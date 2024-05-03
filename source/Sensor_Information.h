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