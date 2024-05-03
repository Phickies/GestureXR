#include <Wire.h>
#include "SparkFun_BMI270_Arduino_Library.h"
#include "Sensor_Information.h"

#define INT_PIN 4
#define SDA_1_PIN 13
#define SCL_1_PIN 12
#define SDA_2_PIN 11
#define SCL_2_PIN 14
#define TOUCH_PIN 1

#define PI 3.14159

#define COM_RATE 400000  // 400KHz
#define DT 0.001         // 1 ms

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
    Quaternion q0, q1, q2, q3;
    Vector3 accleration, rotateSpeed;
    BMI270 imu;
    String name;
  Method:
    void setupI2C(TwoWire* i2cBus, uint8_t sensorAddress);
    void setupConfig(bmi2_sens_config accelConfig, bmi2_sens_config gyroConfig, bmi2_int_pin_config intConfig);
    void calibrateSensor();
    void getRawData();
    void filterDataAHSR();
    void serialPrintRawData();
    void serialPrintQuaternion();
  */
class Finger {
  public:

  volatile float q0 = 1.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;

  struct Vector3 {
    float x, y, z;
  };

  Vector3 acceleration, rotateSpeed;
  String name;
  BMI270 imu;

  Finger(const String& name)
    : name(name), acceleration({ 0, 0, 0 }), rotateSpeed({ 0, 0, 0 }){};

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

  // Filter the data with fancy MadgwickAHSR Algorithm
  void fileDataAHSR(const float sampleFreq, const float beta) {
    this->MadgwickAHRSupdateIMU(
      deg2rad(this->rotateSpeed.x),
      deg2rad(this->rotateSpeed.y),
      deg2rad(this->rotateSpeed.z),
      this->acceleration.x,
      this->acceleration.y,
      this->acceleration.z,
      sampleFreq,
      beta);
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
    Serial.print(this->rotateSpeed.z);
    Serial.println(", ");
  }

  void serialPrintQuaternion() {
    Serial.print(this->q0);
    Serial.print(", ");
    Serial.print(this->q1);
    Serial.print(", ");
    Serial.print(this->q2);
    Serial.print(", ");
    Serial.print(this->q3);
    Serial.print(", ");
  }

  void serialPrintQuaternionEnd() {
    Serial.print(this->q0);
    Serial.print(", ");
    Serial.print(this->q1);
    Serial.print(", ");
    Serial.print(this->q2);
    Serial.print(", ");
    Serial.println(this->q3);
  }

private:
  float deg2rad(float value) {
    return (float)(PI / 180) * value;
  }

  float integrate(float value, float dt) {
    return value * dt;
  }

  void MadgwickAHRSupdateIMU(float gx, float gy, float gz, float ax, float ay, float az, const float sampleFreq, const float beta) {
    float recipNorm;
    float s0, s1, s2, s3;
    float qDot1, qDot2, qDot3, qDot4;
    float _2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2, _8q1, _8q2;
    float q0q0, q1q1, q2q2, q3q3;

    // Rate of change of quaternion from gyroscope
    qDot1 = 0.5f * (-this->q1 * gx - this->q2 * gy - this->q3 * gz);
    qDot2 = 0.5f * (this->q0 * gx + this->q2 * gz - this->q3 * gy);
    qDot3 = 0.5f * (this->q0 * gy - this->q1 * gz + this->q3 * gx);
    qDot4 = 0.5f * (this->q0 * gz + this->q1 * gy - this->q2 * gx);

    // Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
    if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {
      // Normalize accelerometer measurement
      recipNorm = invSqrt(ax * ax + ay * ay + az * az);
      ax *= recipNorm;
      ay *= recipNorm;
      az *= recipNorm;

      // Auxiliary variables to avoid repeated arithmetic
      _2q0 = 2.0f * this->q0;
      _2q1 = 2.0f * this->q1;
      _2q2 = 2.0f * this->q2;
      _2q3 = 2.0f * this->q3;
      _4q0 = 4.0f * this->q0;
      _4q1 = 4.0f * this->q1;
      _4q2 = 4.0f * this->q2;
      _8q1 = 8.0f * this->q1;
      _8q2 = 8.0f * this->q2;
      q0q0 = this->q0 * this->q0;
      q1q1 = this->q1 * this->q1;
      q2q2 = this->q2 * this->q2;
      q3q3 = this->q3 * this->q3;

      // Gradient decent algorithm corrective step
      s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
      s1 = _4q1 * q3q3 - _2q3 * ax + 4.0f * q0q0 * this->q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
      s2 = 4.0f * q0q0 * this->q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
      s3 = 4.0f * q1q1 * this->q3 - _2q1 * ax + 4.0f * q2q2 * this->q3 - _2q2 * ay;
      recipNorm = invSqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);  // normalize step magnitude
      s0 *= recipNorm;
      s1 *= recipNorm;
      s2 *= recipNorm;
      s3 *= recipNorm;

      // Apply feedback step
      qDot1 -= beta * s0;
      qDot2 -= beta * s1;
      qDot3 -= beta * s2;
      qDot4 -= beta * s3;
    }

    // Integrate rate of change of quaternion to yield quaternion
    this->q0 += qDot1 * (1.0f / sampleFreq);
    this->q1 += qDot2 * (1.0f / sampleFreq);
    this->q2 += qDot3 * (1.0f / sampleFreq);
    this->q3 += qDot4 * (1.0f / sampleFreq);

    // Normalize quaternion
    recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    this->q0 *= recipNorm;
    this->q1 *= recipNorm;
    this->q2 *= recipNorm;
    this->q3 *= recipNorm;
  }

  // Fast inverse square-root
  float invSqrt(float x) {
    float halfx = 0.5f * x;
    float y = x;
    long i = *(long*)&y;
    i = 0x5f3759df - (i >> 1);
    y = *(float*)&i;
    y = y * (1.5f - (halfx * y * y));
    return y;
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
    indexFinger.getRawData();
    thumbFinger.getRawData();
    middleFinger.fileDataAHSR(400, 0.1);
    indexFinger.fileDataAHSR(400, 0.1);
    thumbFinger.fileDataAHSR(400, 0.1);
    middleFinger.serialPrintQuaternion();
    indexFinger.serialPrintQuaternion();
    thumbFinger.serialPrintQuaternionEnd();
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