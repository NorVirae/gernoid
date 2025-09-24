// pca9685_single_channel_calibrate.ino
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

const int TEST_CHANNEL = 4;  // change to whatever channel your MG90S is plugged into
const int PWM_FREQ = 50;     // 50 Hz
// Start with these conservative values; widen if needed (but be careful)
int TEST_MIN_US = 500;  // try 1000, then 900, 800... if needed
int TEST_MAX_US = 2500;  // try 2000, then 2100, 2300... if needed

uint16_t usToTick(long us) {
  long period_us = 1000000L / PWM_FREQ;  // 20000 for 50Hz
  long ticks = (us * 4096L) / period_us;
  if (ticks < 0) ticks = 0;
  if (ticks > 4095) ticks = 4095;
  return (uint16_t)ticks;
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  pwm.begin();
  pwm.setPWMFreq(PWM_FREQ);
  delay(10);
  Serial.println("Calibration test: PCA9685 single channel sweep");
  Serial.print("Channel: ");
  Serial.println(TEST_CHANNEL);
  Serial.print("MIN_US: ");
  Serial.println(TEST_MIN_US);
  Serial.print("MAX_US: ");
  Serial.println(TEST_MAX_US);
}

void loop() {
  // Sweep up
  for (int angle = 0; angle <= 180; angle += 5) {
    long us = map(angle, 0, 180, TEST_MIN_US, TEST_MAX_US);
    uint16_t tick = usToTick(us);
    pwm.setPWM(TEST_CHANNEL, 0, tick);
    Serial.print("angle=");
    Serial.print(angle);
    Serial.print("  us=");
    Serial.print(us);
    Serial.print("  tick=");
    Serial.println(tick);
    delay(200);  // slow so you can observe
  }
  delay(500);

  // Sweep down
  for (int angle = 180; angle >= 0; angle -= 5) {
    long us = map(angle, 0, 180, TEST_MIN_US, TEST_MAX_US);
    uint16_t tick = usToTick(us);
    pwm.setPWM(TEST_CHANNEL, 0, tick);
    Serial.print("angle=");
    Serial.print(angle);
    Serial.print("  us=");
    Serial.print(us);
    Serial.print("  tick=");
    Serial.println(tick);
    delay(200);
  }
  delay(1000);
}
