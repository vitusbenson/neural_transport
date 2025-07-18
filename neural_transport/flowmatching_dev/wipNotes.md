Flow Matching Steps

1. Load batch properly (only 'co2massmix' first)
2. Get training_forward running (add offset/scale, dt_alpha/dt_sigma)
3. Get inference_forward runnning
4. Implement time embedding in UNet (also other models?)
5. Integrate into neural_transport
6. test a training
7. Add possibility of covariats and (CO_2)_{t-1}
8. Load OCO-2 dataset into neural_transport
9. Integrate OCO-2 data into carbonbench
10. Extend FlowMatching to use X_0 = noise + weight * OCO-2
11. Handle (OCO-2)_t