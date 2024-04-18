
A interactive controller hand gesture tracker implement Deep Learning for extracting command

go to the library, add this line of code to the getSensorData function:

/// @brief Gets data from the sensor. Must be called to update the data struct
/// @return Error code (0 is success, negative is failure, positive is warning)
int8_t BMI270::getSensorData()
{
    // Variable to track errors returned by API calls
    int8_t err = BMI2_OK;

    // Get raw data from sensor
    bmi2_sens_data rawData;
    err = bmi2_get_sensor_data(&rawData, &sensor);
    if(err != BMI2_OK) return err;

    data.accelX = rawData.acc.x;
    data.accelY = rawData.acc.y;
    data.accelZ = rawData.acc.z;

    data.gyroX = rawData.gyr.x;
    data.gyroY = rawData.gyr.y;
    data.gyroZ = rawData.gyr.z;

    // Convert raw data to g's and deg/sec
    // convertRawData(&rawData, &data);

    return BMI2_OK;
}


After that, go to the header file of the library and change the variable type of <struct BMI270_SensorData> to int16_t

struct BMI270_SensorData
{
    // Acceleration in g's
    int16_t accelX;
    int16_t accelY;
    int16_t accelZ;

    // Rotation in deg/sec
    int16_t gyroX;
    int16_t gyroY;
    int16_t gyroZ;

    // Auxiliary sensor data, raw bytes
    uint8_t auxData[BMI2_AUX_NUM_BYTES];

    // Time of this data in milliseconds, measured by sensor
    uint32_t sensorTimeMillis;
};
