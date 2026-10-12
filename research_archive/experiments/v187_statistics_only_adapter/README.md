# V187 Statistics-Only Adapter

V187 freezes every inherited V23 parameter and trains only the new official-IMU
statistics residual plus its zero-initialized scalar gate.  Batch-normalization
buffers also remain frozen.  The initial network is therefore exactly V23, and
the optimizer cannot trade away visual, skeleton, temporal-IMU, fusion, or
classifier capability while learning the restored magnetometer signal.

