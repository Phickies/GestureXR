#include <Wire.h>
#include "SparkFun_BMI270_Arduino_Library.h"

// The accelerometer and gyroscope can be configured with multiple settings
// to reduce the measurement noise. Both sensors have the following settings
// in common:
// .range       - Measurement range. Lower values give more resolution, but
//                doesn't affect noise significantly, and limits the max
//                measurement before saturating the sensor
// .odr         - Output data rate in Hz. Lower values result in less noise,
//                but lower sampling rates.
// .filter_perf - Filter performance mode. Performance oprtimized mode
//                results in less noise, but increased power consumption
// .bwp         - Filter bandwidth parameter. This has several possible
//                settings that can reduce noise, but cause signal delay
//
// Both sensors have different possible values for each setting:
//
// Accelerometer values:
// .range       - 2g to 16g
// .odr         - Depends on .filter_perf:
//                  Performance mode: 12.5Hz to 1600Hz
//                  Power mode:       0.78Hz to 400Hz
// .bwp         - Depends on .filter_perf:
//                  Performance mode: Normal, OSR2, OSR4, CIC
//                  Power mode:       Averaging from 1 to 128 samples
//
// Gyroscope values:
// .range       - 125dps to 2000dps (deg/sec)
// .ois_range   - 250dps or 2000dps (deg/sec) Only relevant when using OIS,
//                see datasheet for more info. Defaults to 250dps
// .odr         - Depends on .filter_perf:
//                  Performance mode: 25Hz to 3200Hz
//                  Power mode:       25Hz to 100Hz
// .bwp         - Normal, OSR2, OSR4, CIC
// .noise_perf  - Similar to .filter_perf. Performance oprtimized mode
//                results in less noise, but increased power consumption
//
// Note that not all combinations of values are possible. The performance
// mode restricts which ODR settings can be used, and the ODR restricts some
// bandwidth parameters. An error code is returned by setConfig, which can
// be used to determine whether the selected settings are valid.

#define ACC_RANGE BMI2_ACC_RANGE_16G  // 16G
#define ACC_ODR BMI2_ACC_ODR_1600HZ   // 1600Hz
#define ACC_BWP BMI2_ACC_NORMAL_AVG4  // Normal

#define GYRO_RANGE BMI2_GYR_RANGE_2000  // 2000dps
#define GYRO_ODR BMI2_GYR_ODR_3200HZ    // 3200Hz
#define GYRO_BWP BMI2_GYR_NORMAL_MODE   // Normal

#define FILTER_MODE BMI2_PERF_OPT_MODE  // Performance mode

#define INT_PIN 4
#define SDA_1_PIN 13
#define SCL_1_PIN 12
#define SDA_2_PIN 11
#define SCL_2_PIN 14
#define TOUCH_PIN 1

int touch_val = 0;

#define COM_RATE 400000  // 400KHz


// Create a new sensor object
BMI270 imu_middle_f;
BMI270 imu_index_f;
BMI270 imu_thumb_f;

bmi2_sens_config accelConfig;
bmi2_sens_config gyroConfig;

// Setting global variable
unsigned long waitingTimeBeforePowerDown = 30000;  // 30 seconds
unsigned long delayTimeForwaiting = 0;
unsigned long startTime = 0;

// I2C address selection
uint8_t i2cAddress1 = BMI2_I2C_PRIM_ADDR;  // 0x68
uint8_t i2cAddress2 = BMI2_I2C_SEC_ADDR;  // 0x69

// Flag to know when interrupts occur
volatile bool interruptOccurred = false;

TwoWire i2c1 = TwoWire(0);
TwoWire i2c2 = TwoWire(1);



void handleInterrupt() {
  interruptOccurred = true;
}


void setup() {
  // Start serial
  Serial.begin(921600);

  // Initialize the I2C library
  i2c1.begin(SDA_2_PIN, SCL_2_PIN);
  i2c2.begin(SDA_1_PIN, SCL_1_PIN);


  while (imu_middle_f.beginI2C(i2cAddress1, i2c1) != BMI2_OK) {
    Serial.println(imu_middle_f.beginI2C(i2cAddress1, i2c1));
    // Not connected, inform user
    Serial.println("Error: BMI270 in the middle finger is not connected, error code is printed aboved!");

    // Wait a bit to see if connection is established
    delay(1000);
  }
  Serial.println("BMI270 in middle finger is connected!");

  while (imu_index_f.beginI2C(i2cAddress1, i2c2) != BMI2_OK) {
    Serial.println(imu_index_f.beginI2C(i2cAddress1, i2c2));
    // Not connected, inform user
    Serial.println("Error: BMI270 in index finger is not connected, error code is printed aboved!");

    // Wait a bit to see if connection is established
    delay(1000);
  }
  Serial.println("BMI270 in the index finger is connected!");

  while (imu_thumb_f.beginI2C(i2cAddress2, i2c2) != BMI2_OK) {
    Serial.println(imu_thumb_f.beginI2C(i2cAddress2, i2c2));
    // Not connected, inform user
    Serial.println("Error: BMI270 in thumb finger is not connected,  error code is printed aboved!");

    // Wait a bit to see if connection is established
    delay(1000);
  }
  Serial.println("BMI270 in the thumb finger is connected!");

  // Set accelerometer config
  accelConfig.type = BMI2_ACCEL;
  accelConfig.cfg.acc.odr = ACC_ODR;
  accelConfig.cfg.acc.bwp = ACC_BWP;
  accelConfig.cfg.acc.filter_perf = FILTER_MODE;
  accelConfig.cfg.acc.range = ACC_RANGE;
  while (imu_middle_f.setConfig(accelConfig) != BMI2_OK) {
    Serial.println("Accelerometer config for middle fingle not valid!");
    Serial.println(imu_middle_f.setConfig(accelConfig));
    delay(1000);
  }
  while (imu_index_f.setConfig(accelConfig) != BMI2_OK) {
    Serial.println("Accelerometer config for index fingle not valid!");
    Serial.println(imu_index_f.setConfig(accelConfig));
    delay(1000);
  }
  while (imu_thumb_f.setConfig(accelConfig) != BMI2_OK) {
    Serial.println("Accelerometer config for thumb fingle not valid!");
    Serial.println(imu_thumb_f.setConfig(accelConfig));
    delay(1000);
  }

  // Set gyroscope config
  gyroConfig.type = BMI2_GYRO;
  gyroConfig.cfg.gyr.odr = GYRO_ODR;
  gyroConfig.cfg.gyr.bwp = GYRO_BWP;
  gyroConfig.cfg.gyr.filter_perf = FILTER_MODE;
  gyroConfig.cfg.gyr.range = GYRO_RANGE;
  gyroConfig.cfg.gyr.noise_perf = FILTER_MODE;
  while (imu_middle_f.setConfig(gyroConfig) != BMI2_OK) {
    Serial.println("Gyroscope config for middle fingle not valid!");
    Serial.println(imu_middle_f.setConfig(gyroConfig));
    delay(1000);
  }
  while (imu_index_f.setConfig(gyroConfig) != BMI2_OK) {
    Serial.println("Gyroscope config for index fingle not valid!");
    Serial.println(imu_index_f.setConfig(gyroConfig));
    delay(1000);
  }
  while (imu_thumb_f.setConfig(gyroConfig) != BMI2_OK) {
    Serial.println("Gyroscope config for thumb fingle not valid!");
    Serial.println(imu_thumb_f.setConfig(gyroConfig));
    delay(1000);
  }

  Serial.println("Configuration valid!");

  // Initialize interupt pin
  struct bmi2_int_pin_config interruptConfig;

  imu_middle_f.mapInterruptToPin(BMI2_DRDY_INT, BMI2_INT1);
  imu_index_f.mapInterruptToPin(BMI2_DRDY_INT, BMI2_INT1);
  imu_thumb_f.mapInterruptToPin(BMI2_DRDY_INT, BMI2_INT1);

  interruptConfig.pin_type = BMI2_INT1;
  interruptConfig.int_latch = BMI2_INT_NON_LATCH;
  interruptConfig.pin_cfg[0].output_en = BMI2_INT_OUTPUT_ENABLE;
  interruptConfig.pin_cfg[0].od = BMI2_INT_PUSH_PULL;
  interruptConfig.pin_cfg[0].lvl = BMI2_INT_ACTIVE_LOW;
  interruptConfig.pin_cfg[0].input_en = BMI2_INT_INPUT_DISABLE;

  imu_middle_f.setInterruptPinConfig(interruptConfig);
  imu_index_f.setInterruptPinConfig(interruptConfig);
  imu_thumb_f.setInterruptPinConfig(interruptConfig);

  attachInterrupt(INT_PIN, handleInterrupt, RISING);

  Serial.println("Settup interuptpin valid!");

  // Calibration
  Serial.println("Place the sensor on a flat surface and leave it stationary.");
  // Serial.println("Enter any key to begin calibration.");

  // // Throw away any previous inputs
  // while (Serial.available() != 0) { Serial.read(); }
  // // Wait for user input
  // while (Serial.available() == 0) {}

  // Perform component retrim for the gyroscope. According to the datasheet,
  // the gyroscope has a typical error of 2%, but running the CRT can reduce
  // that error to 0.4%
  Serial.println("Performing component retrimming...");
  imu_middle_f.performComponentRetrim();
  imu_index_f.performComponentRetrim();
  imu_thumb_f.performComponentRetrim();

  // Perform offset calibration for both the accelerometer and IMU. This will
  // automatically determine the offset of each axis of each sensor, and
  // that offset will be subtracted from future measurements. Note that the
  // offset resolution is limited for each sensor:
  //
  // Accelerometer offset resolution: 0.0039 g
  // Gyroscope offset resolution: 0.061 deg/sec
  Serial.println("Performing acclerometer offset calibration...");
  imu_middle_f.performAccelOffsetCalibration(BMI2_GRAVITY_POS_Z);
  imu_index_f.performAccelOffsetCalibration(BMI2_GRAVITY_POS_Z);
  imu_thumb_f.performAccelOffsetCalibration(BMI2_GRAVITY_POS_Z);
  Serial.println("Performing gyroscope offset calibration...");
  imu_middle_f.performGyroOffsetCalibration();
  imu_index_f.performGyroOffsetCalibration();
  imu_thumb_f.performGyroOffsetCalibration();

  // // Throw away any previous inputs
  // while (Serial.available() != 0) { Serial.read(); }
  // // Wait for user input
  // while (Serial.available() == 0) {}

  // // Check to see if user wants to save values to NVM
  // if (Serial.read() == 'Y') {
  //   Serial.println();
  //   Serial.println("!!! WARNING !!! WARNING !!! WARNING !!! WARNING !!! WARNING !!!");
  //   Serial.println();
  //   Serial.println("The BMI270's NVM only supports 14 write cycles TOTAL!");
  //   Serial.println("Are you sure you want to save to the NVM? If so, enter 'Y' again");

  //   // Throw away any previous inputs
  //   while (Serial.available() != 0) { Serial.read(); }
  //   // Wait for user input
  //   while (Serial.available() == 0) {}

  //   // Check to see if user really wants to save values to NVM
  //   if (Serial.read() == 'Y') {
  //     // Save NVM contents
  //     int8_t err = imu_middle_f.saveNVM();

  //     // Check to see if the NVM saved successfully
  //     if (err == BMI2_OK) {
  //       Serial.println();
  //       Serial.println("Calibration values have been saved to the NVM!");
  //     } else {
  //       Serial.print("Error saving to NVM, error code: ");
  //       Serial.println(err);
  //     }
  //   }
  // }
  Serial.println("Calibration done! Start collecting data!");

  delay(1);

  // Setting up the waitingTime
  delayTimeForwaiting = esp_timer_get_time();

  pinMode(TOUCH_PIN, INPUT);
}

void loop() {

  // Update timer
  startTime = esp_timer_get_time();

  touch_val = digitalRead(TOUCH_PIN);

  if (touch_val == 1) {
    while (imu_middle_f.getSensorData() != BMI2_OK) {
      Serial.println(imu_middle_f.getSensorData());
      Serial.print("Error collecting data on middle finger");
    }
    // Print acceleration data
    // Serial.print("AX: ");
    Serial.print(imu_middle_f.data.accelX);
    Serial.print(", ");
    Serial.print(imu_middle_f.data.accelY);
    Serial.print(", ");
    Serial.print(imu_middle_f.data.accelZ);

    // Print rotation data
    Serial.print(", ");
    Serial.print(imu_middle_f.data.gyroX);
    Serial.print(", ");
    Serial.print(imu_middle_f.data.gyroY);
    Serial.print(", ");
    Serial.print(imu_middle_f.data.gyroZ);

    while (imu_index_f.getSensorData() != BMI2_OK) {
      Serial.println(imu_index_f.getSensorData());
      Serial.print("Error collecting data on index finger");
    }
    // Print acceleration data
    Serial.print(", ");
    Serial.print(imu_index_f.data.accelX);
    Serial.print(", ");
    Serial.print(imu_index_f.data.accelY);
    Serial.print(", ");
    Serial.print(imu_index_f.data.accelZ);

    // Print rotation data
    Serial.print(", ");
    Serial.print(imu_index_f.data.gyroX);
    Serial.print(", ");
    Serial.print(imu_index_f.data.gyroY);
    Serial.print(", ");
    Serial.print(imu_index_f.data.gyroZ);

    while (imu_thumb_f.getSensorData() != BMI2_OK) {
      Serial.println(imu_thumb_f.getSensorData());
      Serial.print("Error collecting data on thumb finger");
    }
    Serial.print(", ");
    Serial.print(imu_thumb_f.data.accelX);
    Serial.print(", ");
    Serial.print(imu_thumb_f.data.accelY);
    Serial.print(", ");
    Serial.print(imu_thumb_f.data.accelZ);

    // Print rotation data
    Serial.print(", ");
    Serial.print(imu_thumb_f.data.gyroX);
    Serial.print(", ");
    Serial.print(imu_thumb_f.data.gyroY);
    Serial.print(", ");
    Serial.println(imu_thumb_f.data.gyroZ);

    delayTimeForwaiting = esp_timer_get_time();
    // }

    // Enter saving power after time
    if (delayTimeForwaiting - startTime > waitingTimeBeforePowerDown) {
      // do something
    }
  }

  // Get measurements from the sensor. This must be called before accessing
  // the sensor data, otherwise it will never update
  // if (interruptOccurred) {

  // interruptOccurred = false;
}