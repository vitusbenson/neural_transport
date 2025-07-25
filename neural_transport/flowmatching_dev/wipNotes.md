Flow Matching Steps

01. Load batch properly (only 'co2massmix' first)                        v
02. Get training_forward running (add offset/scale, dt_alpha/dt_sigma)   v
03. Get inference_forward runnning                                       v
04. Implement time embedding in UNet (a)                                 v
05. Integrate into neural_transport                                      
06. test a training                                                      
07. create slurm script                                                  v                        
08. Add possibility of covariats and (CO_2)_{t-1}                        
09. Load OCO-2 dataset into neural_transport                             
10. Integrate OCO-2 data into carbonbench                                
11. Extend FlowMatching to use X_0 = noise + weight * OCO-2             
12. Handle (OCO-2)_t                                                    
13. Attack with Ruff

(a): It is just stacked as another channel:
        `x_in = torch.cat(list(batch_normalized.values()), dim=-1)`