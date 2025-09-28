# solady-vanity

CUDA-based vanity searcher for Solady CREATE3 deployments. The program scans
CREATE3 salts so the resulting contract address matches a user-supplied hex
prefix. Build and run with the helper script:

```
./run.sh \
  --deployer 0xBA203fFDB6727c59e31D73d66290fFb47728e4Cb \
  --init-hash 0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f \
  --prefix d0000000
```

The console status now also reports the runtime, current hash rate, and the
expected time to find a match at the observed rate. On success, the tool prints
the 32-byte salt, checksum-encoded CREATE3 address, elapsed time, and the
expected find time at the recorded hash rate.
