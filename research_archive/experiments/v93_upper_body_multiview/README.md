# V93 Upper-Body Multi-View

V93 tests a fixed semantic crop motivated by the V91 error audit. The existing
160-pixel V88 model sees both its standard full-person TTA view and the upper
72% of the same person crop. The two view families receive equal probability
weight, with no learned parameters and no searched crop ratio or blend weight.

The 72% boundary is inherited unchanged from the repository's earlier
label-independent upper-body preprocessing definition. Training and pseudo
adaptation remain identical to V88; only inference views change.
