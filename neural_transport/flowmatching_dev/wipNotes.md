Flow Matching Steps

1. Load batch properly (only 'co2massmix' first)                        v
2. Get training_forward running (add offset/scale, dt_alpha/dt_sigma)   v
3. Get inference_forward runnning                                       v
4. Implement time embedding in UNet (a)                                 v
5. Integrate into neural_transport                                      x
6. test a training                                                      x
7. Add possibility of covariats and (CO_2)_{t-1}                        x
8. Load OCO-2 dataset into neural_transport                             x
9. Integrate OCO-2 data into carbonbench                                x
10. Extend FlowMatching to use X_0 = noise + weight * OCO-2             x
11. Handle (OCO-2)_t                                                    x


(a): It is just stacked as another channel:
        `x_in = torch.cat(list(batch_normalized.values()), dim=-1)`
