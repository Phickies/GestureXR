#include <Wire.h>
#include "SparkFun_BMI270_Arduino_Library.h"
#include "Sensor_Information.h"

#define INT_PIN 4
#define SDA_1_PIN 13
#define SCL_1_PIN 12
#define SDA_2_PIN 11
#define SCL_2_PIN 14
#define TOUCH_PIN 1

#define COM_RATE 400000  // 400KHz
#define DT 0.1           // 100 ms

#define ACC_RANGE BMI2_ACC_RANGE_16G  // 16G
#define ACC_ODR BMI2_ACC_ODR_1600HZ   // 1600Hz
#define ACC_BWP BMI2_ACC_NORMAL_AVG4  // Normal

#define GYRO_RANGE BMI2_GYR_RANGE_2000  // 2000dps
#define GYRO_ODR BMI2_GYR_ODR_3200HZ    // 3200Hz
#define GYRO_BWP BMI2_GYR_NORMAL_MODE   // Normal

#define FILTER_MODE BMI2_PERF_OPT_MODE  // Performance mode

/* 
  Class finger to abstract IMU and data in each finger
  Attribute:
    Vector3 accleration, velocity, position, rotateSpeed;
    BMI270 imu;
    String name;
  Method:
    void setupI2C(TwoWire* i2cBus, uint8_t sensorAddress);
    void setupConfig(bmi2_sens_config accelConfig, bmi2_sens_config gyroConfig, bmi2_int_pin_config intConfig);
    void calibrateSensor();
    void getRawData();
    void serialPrintRawData();
  */
class Finger {
public:
  struct Vector3 {
    float x, y, z;
  };

  String name;
  Vector3 acceleration, velocity, position, rotateSpeed;
  BMI270 imu;

  Finger(const String& name)
    : name(name), acceleration({ 0, 0, 0 }), velocity({ 0, 0, 0 }), position({ 0, 0, 0 }), rotateSpeed({ 0, 0, 0 }){};

  // Intergrate gyroValue
  float intergrateGyroValue(int16_t gyroValue, float dt) {
    return gyroValue * dt;
  }

  // Intergarte accelValue
  float intergrateAccelValue(int16_t accelValue, float dt) {
    return accelValue * dt;
  }

  void setPosition() {
  }

  // Return the Vector3 position value (x, y, z)
  Vector3 getPosition() {
    return this->position;
  }

  // Return the Vector3 velocity value (x, y, z)
  Vector3 getVelocity() {
    return this->velocity;
  }

  // Setup I2C port and establish connection with the IMU sensor
  void setupI2C(TwoWire* i2cBus, uint8_t sensorAddress) {
    uint8_t result = this->imu.beginI2C(BMI2_I2C_PRIM_ADDR, *i2cBus);
    while (result != BMI2_OK) {
      Serial.print("Error for ");
      Serial.print(this->name);
      Serial.print(" ");
      Serial.println(result);
      delay(1000);
    }
  }

  // Setup config for IMU sensor including acceleration, gyroscope and Interrupt pin
  void setupConfig(bmi2_sens_config accelConfig, bmi2_sens_config gyroConfig, bmi2_int_pin_config intConfig) {
    uint8_t resultAccel = this->imu.setConfig(accelConfig);
    uint8_t resultGyro = this->imu.setConfig(gyroConfig);
    while (resultAccel != BMI2_OK) {
      Serial.print("Error accel config for ");
      Serial.print(this->name);
      Serial.print(" ");
      Serial.println(resultAccel);
      delay(1000);
    }
    while (resultGyro != BMI2_OK) {
      Serial.print("Error gyro config for ");
      Serial.print(this->name);
      Serial.print(" ");
      Serial.println(resultGyro);
      delay(1000);
    }
    this->imu.mapInterruptToPin(BMI2_DRDY_INT, BMI2_INT1);
    this->imu.setInterruptPinConfig(intConfig);
  }

  /* 
    Calibration data

    Perform component retrim for the gyroscope. According to the datasheet,
    the gyroscope has a typical error of 2%, but running the CRT can reduce
    that error to 0.4%

    Perform offset calibration for both the accelerometer and IMU. This will
    automatically determine the offset of each axis of each sensor, and
    that offset will be subtracted from future measurements. Note that the
    offset resolution is limited for each sensor:
    
    Accelerometer offset resolution: 0.0039 g
    Gyroscope offset resolution: 0.061 deg/sec
  */
  void calibrateSensor() {
    Serial.print("Performing component retrimming for ");
    Serial.println(this->name);
    this->imu.performComponentRetrim();
    Serial.print("Performing acclerometer offset calibration for ");
    Serial.println(this->name);
    this->imu.performAccelOffsetCalibration(BMI2_GRAVITY_POS_Z);
    Serial.print("Performing gyroscope offset calibration for ");
    Serial.println(this->name);
    this->imu.performGyroOffsetCalibration();
  }

  // Get the raw data for accelaration and gyro from the library API
  void getRawData() {
    uint8_t result = this->imu.getSensorData();
    while (result != BMI2_OK) {
      Serial.print("Error collecting data on ");
      Serial.print(this->name);
      Serial.print(" ");
      Serial.println(result);
    }
    this->acceleration.x = this->imu.data.accelX;
    this->acceleration.y = this->imu.data.accelY;
    this->acceleration.z = this->imu.data.accelZ;

    this->rotateSpeed.x = this->imu.data.gyroX;
    this->rotateSpeed.y = this->imu.data.gyroY;
    this->rotateSpeed.z = this->imu.data.gyroZ;
  }

  // Serial print raw acceleration and gyroscope
  void serialPrintRawData() {
    Serial.print(this->acceleration.x);
    Serial.print(", ");
    Serial.print(this->acceleration.y);
    Serial.print(", ");
    Serial.print(this->acceleration.z);
    Serial.print(", ");
    Serial.print(this->rotateSpeed.x);
    Serial.print(", ");
    Serial.print(this->rotateSpeed.y);
    Serial.print(", ");
    Serial.println(this->rotateSpeed.z);
  }
};

//-----------------------------------------
//          Main function
//-----------------------------------------

struct bmi2_sens_config accelConfig;
struct bmi2_sens_config gyroConfig;
struct bmi2_int_pin_config interruptConfig;

// Flag to know when interrupts occur
volatile bool interruptOccurred = false;
uint8_t touchVal = 0;

TwoWire i2cBus1 = TwoWire(0);
TwoWire i2cBus2 = TwoWire(1);

Finger middleFinger("Middle");
Finger indexFinger("Index");
Finger thumbFinger("Thumb");

void setup() {

  // Start serial
  Serial.begin(921600);
  Serial.println("GestureXR start");

  // Initialize the I2C library
  i2cBus1.begin(SDA_2_PIN, SCL_2_PIN);
  i2cBus2.begin(SDA_1_PIN, SCL_1_PIN);

  // Setup I2C connection
  middleFinger.setupI2C(&i2cBus1, BMI2_I2C_PRIM_ADDR);
  indexFinger.setupI2C(&i2cBus2, BMI2_I2C_PRIM_ADDR);
  thumbFinger.setupI2C(&i2cBus1, BMI2_I2C_SEC_ADDR);

  // Setup Config
  accelConfig = setConfigAccel(&accelConfig);
  gyroConfig = setConfigGyro(&gyroConfig);
  interruptConfig = setConfigInterupt(&interruptConfig);
  middleFinger.setupConfig(accelConfig, gyroConfig, interruptConfig);
  indexFinger.setupConfig(accelConfig, gyroConfig, interruptConfig);
  thumbFinger.setupConfig(accelConfig, gyroConfig, interruptConfig);
  attachInterrupt(INT_PIN, handleInterrupt, RISING);

  // Calibrating Sensor
  Serial.println("Place the sensor on a flat surface and leave it stationary.");
  middleFinger.calibrateSensor();
  indexFinger.calibrateSensor();
  thumbFinger.calibrateSensor();
  Serial.println("Calibration done! Start collecting data!");

  // Delay to setup interrupt pin
  delay(100);
  pinMode(TOUCH_PIN, INPUT);
}

void loop() {

  touchVal = digitalRead(TOUCH_PIN);

  if (touchVal) {
    middleFinger.getRawData();
    // indexFinger.getRawData();
    // thumbFinger.getRawData();
    middleFinger.serialPrintRawData();
    // indexFinger.serialPrintRawData();
    // thumbFinger.serialPrintRawData();
    delay(DT * 1000);
  }
}

//-----------------------------------------
//          Helper function
//-----------------------------------------

void handleInterrupt() {
  interruptOccurred = true;
}

// Setup configuration for Accel Sensor
bmi2_sens_config& setConfigAccel(bmi2_sens_config* accelConfig) {
  accelConfig->type = BMI2_ACCEL;
  accelConfig->cfg.acc.odr = ACC_ODR;
  accelConfig->cfg.acc.bwp = ACC_BWP;
  accelConfig->cfg.acc.filter_perf = FILTER_MODE;
  accelConfig->cfg.acc.range = ACC_RANGE;
  return *accelConfig;
}

// Setup configuration for Gyro Sensor
bmi2_sens_config& setConfigGyro(bmi2_sens_config* gyroConfig) {
  gyroConfig->type = BMI2_GYRO;
  gyroConfig->cfg.gyr.odr = GYRO_ODR;
  gyroConfig->cfg.gyr.bwp = GYRO_BWP;
  gyroConfig->cfg.gyr.filter_perf = FILTER_MODE;
  gyroConfig->cfg.gyr.range = GYRO_RANGE;
  gyroConfig->cfg.gyr.noise_perf = FILTER_MODE;
  return *gyroConfig;
}

// Setup configuration for Interupt Pin Sensor
bmi2_int_pin_config& setConfigInterupt(bmi2_int_pin_config* intConfig) {
  intConfig->pin_type = BMI2_INT1;
  intConfig->int_latch = BMI2_INT_NON_LATCH;
  intConfig->pin_cfg[0].output_en = BMI2_INT_OUTPUT_ENABLE;
  intConfig->pin_cfg[0].od = BMI2_INT_PUSH_PULL;
  intConfig->pin_cfg[0].lvl = BMI2_INT_ACTIVE_LOW;
  intConfig->pin_cfg[0].input_en = BMI2_INT_INPUT_DISABLE;
  return *intConfig;
}