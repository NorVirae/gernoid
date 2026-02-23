// PCA9685_serial_control_debug.ino
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(); // default addr 0x40

const int NUM_SERVOS = 5;
const uint8_t CHANNELS[NUM_SERVOS] = {0,1,2,3,4}; // physical PCA9685 channels

const int PWM_FREQ = 50;        // Hz
const int SERVOMIN_US = 500;    // MG90S calibrated min
const int SERVOMAX_US = 2500;   // MG90S calibrated max

// Smoothing parameters
const unsigned long STEP_INTERVAL_MS = 20;   // how often (ms) to step toward target
const float STEP_DEGREES = 2.5f;             // max degrees to move per step

String recvBuf = "";

// per-servo state
float currentAngle[NUM_SERVOS];
float targetAngle[NUM_SERVOS];

// per-channel inversion (true -> 180-angle)
bool invertChannel[NUM_SERVOS] = { false, false, false, false, false };

// mapping: input index -> servo index (so you can reorder inputs to match physical servos)
int mapping[NUM_SERVOS] = { 0, 1, 2, 3, 4 };

bool debugMode = false;

uint16_t angleToTick(float angle){
  angle = constrain(angle, 0.0f, 180.0f);
  float us = SERVOMIN_US + (angle / 180.0f) * (SERVOMAX_US - SERVOMIN_US);
  long period_us = 1000000L / PWM_FREQ; // e.g. 20000
  long ticks = (long)((us * 4096.0f) / period_us + 0.5f);
  ticks = constrain(ticks, 0L, 4095L);
  return (uint16_t)ticks;
}

void applyAllCurrentAnglesToDriver() {
  for (int i = 0; i < NUM_SERVOS; ++i) {
    uint16_t tick = angleToTick(currentAngle[i]);
    pwm.setPWM(CHANNELS[i], 0, tick);
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(PWM_FREQ);
  delay(10);

  for (int i = 0; i < NUM_SERVOS; ++i) {
    currentAngle[i] = 0.0f;
    targetAngle[i] = 0.0f;
  }
  applyAllCurrentAnglesToDriver();

  Serial.println("PCA9685 debug control ready");
  Serial.println("Send CSV frames:  a0,a1,a2,a3,a4\\n (input order)");
  Serial.println("Commands: INV,1,0,0,1,0  |  MAP,0,1,2,3,4  |  DBG,1  |  TEST");
}

// step ramping
unsigned long lastStepMs = 0;
void stepTowardsTargets() {
  unsigned long now = millis();
  if (now - lastStepMs < STEP_INTERVAL_MS) return;
  lastStepMs = now;

  bool changed = false;
  for (int i = 0; i < NUM_SERVOS; ++i) {
    float diff = targetAngle[i] - currentAngle[i];
    if (fabs(diff) <= 0.01f) continue;
    changed = true;
    if (diff > 0) currentAngle[i] += min(diff, STEP_DEGREES);
    else currentAngle[i] -= min(fabs(diff), STEP_DEGREES);
    currentAngle[i] = constrain(currentAngle[i], 0.0f, 180.0f);
  }
  if (changed) applyAllCurrentAnglesToDriver();
}

// set invert flags from array
void setInvertFromArray(int vals[], int count) {
  if (count != NUM_SERVOS) {
    Serial.println("ERR:INV_count");
    return;
  }
  for (int i=0;i<NUM_SERVOS;++i) invertChannel[i] = (vals[i] != 0);
  Serial.print("INV_SET:");
  for (int i=0;i<NUM_SERVOS;++i){
    Serial.print(invertChannel[i] ? '1' : '0');
    if (i < NUM_SERVOS-1) Serial.print(',');
  }
  Serial.println();
}

// parse INV command e.g. "INV,1,0,0,1,0"
void parseInvertCommand(const String &s) {
  int start = 3;
  while (start < s.length() && (s.charAt(start) == ':' || s.charAt(start) == ',' || s.charAt(start) == ' ')) start++;
  int vals[NUM_SERVOS]; int idx = 0; int tokStart = start;
  for (int i = start; i < s.length() && idx < NUM_SERVOS; ++i) {
    if (s.charAt(i) == ',' || s.charAt(i) == ' ' || s.charAt(i) == ':' ) {
      String piece = s.substring(tokStart, i);
      if (piece.length()) vals[idx++] = piece.toInt();
      tokStart = i + 1;
    }
  }
  if (tokStart < s.length() && idx < NUM_SERVOS) {
    String piece = s.substring(tokStart);
    if (piece.length()) vals[idx++] = piece.toInt();
  }
  if (idx == NUM_SERVOS) setInvertFromArray(vals, idx);
  else Serial.println("ERR:INV_format");
}

// parse MAP command e.g. "MAP,0,1,2,3,4"
void parseMapCommand(const String &s) {
  int start = 3;
  while (start < s.length() && (s.charAt(start) == ':' || s.charAt(start) == ',' || s.charAt(start) == ' ')) start++;
  int vals[NUM_SERVOS]; int idx = 0; int tokStart = start;
  for (int i = start; i < s.length() && idx < NUM_SERVOS; ++i) {
    if (s.charAt(i) == ',' || s.charAt(i) == ' ' || s.charAt(i) == ':' ) {
      String piece = s.substring(tokStart, i);
      if (piece.length()) vals[idx++] = piece.toInt();
      tokStart = i + 1;
    }
  }
  if (tokStart < s.length() && idx < NUM_SERVOS) {
    String piece = s.substring(tokStart);
    if (piece.length()) vals[idx++] = piece.toInt();
  }
  if (idx == NUM_SERVOS) {
    for (int i=0;i<NUM_SERVOS;++i) {
      if (vals[i] >= 0 && vals[i] < NUM_SERVOS) mapping[i] = vals[i];
      else {
        Serial.println("ERR:MAP_range");
        return;
      }
    }
    Serial.print("MAP_SET:");
    for (int i=0;i<NUM_SERVOS;++i) {
      Serial.print(mapping[i]);
      if (i < NUM_SERVOS-1) Serial.print(',');
    }
    Serial.println();
  } else {
    Serial.println("ERR:MAP_format");
  }
}

// parse incoming CSV of 5 values (in input order)
void parseAndSetTargets(const String &s) {
  int inVals[NUM_SERVOS];
  int idx = 0; int start = 0; int len = s.length();
  for (int i = 0; i < len && idx < NUM_SERVOS; ++i) {
    if (s.charAt(i) == ',') {
      String piece = s.substring(start, i);
      inVals[idx++] = piece.toInt();
      start = i + 1;
    }
  }
  if (idx < NUM_SERVOS && start < len) {
    String piece = s.substring(start);
    inVals[idx++] = piece.toInt();
  }

  if (idx == NUM_SERVOS) {
    // apply mapping+inversion: input index i -> servo mapping[i]
    for (int i=0;i<NUM_SERVOS;++i) {
      int servoIdx = mapping[i];
      int raw = constrain(inVals[i], 0, 180);
      float applied = invertChannel[servoIdx] ? (180.0f - (float)raw) : (float)raw;
      targetAngle[servoIdx] = applied;
    }
    if (debugMode) {
      Serial.print("IN: ");
      for (int i=0;i<NUM_SERVOS;++i) { Serial.print(inVals[i]); if (i<NUM_SERVOS-1) Serial.print(','); }
      Serial.print("  -> TARGETS: ");
      for (int i=0;i<NUM_SERVOS;++i) { Serial.print(targetAngle[i]); if (i<NUM_SERVOS-1) Serial.print(','); }
      Serial.print("  TICKS: ");
      for (int i=0;i<NUM_SERVOS;++i) { Serial.print(angleToTick(targetAngle[i])); if (i<NUM_SERVOS-1) Serial.print(','); }
      Serial.println();
    }
    Serial.println("OK");
  } else {
    Serial.println("ERR:count");
  }
}

// Run quick visual test: 0 -> 90 -> 180 for each servo (slowly)
void runVisualTest() {
  Serial.println("TEST_START");
  for (int phase = 0; phase < 3; ++phase) {
    float val = (phase==0)?0.0f : (phase==1)?90.0f : 180.0f;
    for (int i=0;i<NUM_SERVOS;++i) {
      targetAngle[i] = val;
      currentAngle[i] = val; // immediate set to avoid long ramps for test
    }
    applyAllCurrentAnglesToDriver();
    Serial.print("TEST_PHASE: "); Serial.println(val);
    delay(1000);
  }
  Serial.println("TEST_DONE");
}

void loop() {
  // read serial
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      recvBuf.trim();
      if (recvBuf.length()) {
        if (recvBuf.startsWith("INV") || recvBuf.startsWith("inv")) {
          parseInvertCommand(recvBuf);
        } else if (recvBuf.startsWith("MAP") || recvBuf.startsWith("map")) {
          parseMapCommand(recvBuf);
        } else if (recvBuf.startsWith("DBG") || recvBuf.startsWith("dbg")) {
          // DBG,1 or DBG,0
          int val = recvBuf.substring(4).toInt();
          debugMode = (val != 0);
          Serial.print("DBG="); Serial.println(debugMode ? "1":"0");
        } else if (recvBuf.startsWith("TEST") || recvBuf.startsWith("test")) {
          runVisualTest();
        } else {
          parseAndSetTargets(recvBuf);
        }
      }
      recvBuf = "";
    } else {
      recvBuf += c;
      if (recvBuf.length() > 300) recvBuf = ""; // safety
    }
  }

  stepTowardsTargets();
}
