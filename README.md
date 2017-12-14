Pytorch element-wise operation optimization benchmark
=======

### 1. Abstract
Providing a benchmark for element-wise operation performance evaluation on CPU.  

Tested CPU： 

|CPU Model|Sockets|Cores/Socket|Frequency|
|---|---|---|---|
|Intel(R) Xeon(R) CPU E5-2699 v4   |2|22|2.20GHz|
|Intel(R) Xeon(R) Platinum 8180 CPU|2|28|2.50GHz|
|Intel(R) Core(TM) i7-5960X CPU |1|8|3.00GHz|
  
Tested operations:

| | | | | | | |
|:---|:---|:---|:---|:---|:---|:---|
|copy|add|div|sin|exp|sum|prod|   
  
Conclusions:   

* OpenMP threshold which is set to 100k in official version is too high for contiguous tensors of small and medium size to benefit from OpenMP parallelism.
*	Discontiguous tensors' operations can be boosted significantly by __Intel Pytorch__ .
*	The optimal OpenMP threshold is dependent on the operation type and CPU type.  
    - OpenMP threshold becomes smaller for more complex operations.
    - OpenMP threshold of discontiguous tensor is usually lower than that of contiguous tensor.  

__annotation__:  
OpenMP threshold -- If the size of a tensor is larger than the value, the operations run in parallel, otherwise in serial.

This benchmark also gives a rough estimation of optimal OpenMP threshold of copy, add, div, exp, sin, sum and prod operation on different types of CPU.  
  
For contiguous tensor operation:

|   |Xeon(R) Platinum 8180 CPU|Xeon(R) CPU E5-2699 v4| i7-5960X CPU| 
|---|------------------------:|---------------------:|------------:|  
|copy|80k|20k|8k|            
|add |80k|20k|8k|          
|div |50k|10k|2k|          
|exp |1k |1k |1k|          
|sin |1k |1k |1k|        
|sum |1k |1k |1k|            
|prod|1k |1k |1k|           

For discontiguous tensor operation:    
  
||Xeon(R) Platinum 8180 CPU|Xeon(R) CPU E5-2699 v4| i7-5960X CPU|  
|---|------------------------:|---------------------:|------------:|   
|copy|20k|8k |2k|          
|add |20k|8k |2k|         
|div |10k|8k |1k|          
|exp |1k |1k |1k|          
|sin |2k |2k |1k|        
|sum |1k |1k |1k|        
|prod|1k |1k |1k|           
 
   

### 2. Major work
-	Optimal OpenMP threshold is identified to fully exploit performance potentiality on CPU  
The OpenMP threshold of official Pytorch is set to 100K. However, the evidence gained by benchmarking copy, add, div, exp, sin operation in both contiguous and discontiguous cases on different CPU types shows that the value is too high. A rough estimation of optimal OpenMP threshold is also proposed for those operations.
- Discontiguous tensors' operation parallelization with OpenMP   
Slice operation of tensor is very common in science computation. Using slice operation will generate discontiguous tensor. Meanwhile, [Official Pytorch](https://github.com/pytorch/pytorch) does not support parallelism of discontiguous tensor at the moment. Our main work is trying to fill this blank.  Code available at [__dev-omp__](https://github.com/intel/pytorch/tree/dev-omp) and upstreaming is in progress.

  
### 3. Installation and test
#### 3.1 Installation
##### Official Pytorch   
Please refer to official [__link__](https://github.com/pytorch/pytorch)  
##### Intel Pytorch   
Download Intel pytorch source code.
```bash
git clone --recursive -b dev-omp2 https://github.com/intel/pytorch.git
```
Before installing, you should set the CMAKE_PREFIX_PATH.
```
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]
```
Install intel Pytorch
```
python setup.py install
```


#### 3.2 Test

```bash
python benchmark.py <CONTIGUITY> <OPERATION> [OUTPUT FILENAME] 
```
Positional arguments:     
`CONTIUITY`—— operands' contiguity, ontiguous/discontiguous  
`OPERATION`—— operation, copy/add/div/sin/exp/sum/prod 

Optional arguments:  
`o output filename`——output filename, output.log is in default  


### 4. The benchmark result

#### 4.1 Contiguous Tensor Operation OpenMP Threshold Tuning
Add, exp operation for contiguous tensors whose sizes range from 1K to 100K are listed here as test cases. We compiled two versions of official Pytorch by setting two different OpenMP [threshold](https://github.com/pytorch/pytorch/blob/master/aten/src/TH/generic/THTensorMath.c#L13). The threshold of one version is set to 100K to make all of the test case runs in series. Meanwhile the threshold of the other one is set to 800 to make all of the test case in parallel. 

Platform: Platinum 8180  
Operation: add  
Tensor Continuity: contiguous    
Unit: microsecond    

Time cost result is below:  

|Tensor Size|In series|In parallel|SpeedUp|
|---|---:|---:|---:|
|1k	|1.04	|5.15|		0.20X      |
|2k	|1.23	|5.47|		0.22X      |
|3k	|1.33	|5.34|		0.24X      |
|4k	|1.47	|5.41|		0.27X      |
|5k	|1.48	|5.40|		0.27X      |
|8k	|1.81	|5.55|		0.32X      |
|10k|1.98	|5.66|		0.35X      |
|20k|2.74	|6.74|		0.40X      |
|50k|5.12	|6.59|		0.77X      |
|__80k__|__14.79__|__6.59__|		__2.24X__      |
|__100k__|__21.97__|__6.70__|		__3.27X__      |

Conclusion: Setting the threshold to __80K__ is good for __add operation__ of contiguous tensors. 

Platform: Platinum 8180  
Operation: exp  
Tensor Continuity: contiguous    
Unit: microsecond    

Time cost result is below:  

|Tensor Size|In series|In parallel|SpeedUp|
|---|---:|---:|---:|
|__1k__	|__9.48__	|__5.66__|		__1.67X__      |
|__2k__	|__17.00__	|__6.35__|		__2.67X__      |
|__3k__	|__24.82__	|__6.03__|		__4.11X__      |
|__4k__	|__32.52__	|__6.28__|		__5.17X__      |
|__5k__	|__40.33__	|__6.27__|		__6.42X__      |
|__8k__	|__63.58__	|__7.04__|		__9.02X__      |
|__10k__|__79.13__	|__7.61__|		__10.38X__      |
|__20k__|__156.78__	|__9.11__|		__17.20X__      |
|__50k__|__387.85__	|__15.07__|		__25.73X__      |
|__80k__|__623.34__ |__20.23__|		__30.80X__      |
|__100k__|__779.95__|__23.57__|		__33.08X__      |

Conclusion: Setting the threshold to __1K__ is good for __exponential operation__ of contiguous tensors.  

From above results, it is easy to understand that,

- Different operations have their own optimal OpenMP threshold, but 100K is not suitable.
- OpenMP threshold becomes smaller for more complex operations.
  
We don't list all the detailed data for div, sin, sum and prod operation but provide a rough estimation of optimal OpenMP threshold for different operations.  

#### 4.2 Discontiguous tensor operation parallelization 
Add and exp operation performance for discontiguous tensors whose sizes range from 1k to 180k are listed. Official pytorch does not optimize operations for discontiguous tensors with OpenMP but Intel version does. In order to expalin that OpenMP also do good in discontiguous tensor operations and to find a optimal OpenMP threshold, we compiled two versions of Pytorch. One is the Official Pytorch. The other one is the Intel one whose OpenMP threshold is set to 800 to make all test cases run in parallel.  

Platform: Platinum 8180  
Operation: add  
Tensor Continuity: discontiguous    
Unit: microsecond   

Time cost result is below:  

|Tensor Size|In series|In parallel|SpeedUp|
|---|---:|---:|---:|
|1k|	1.69    |	6.98	|	0.24X |
|2k|	2.42    |	7.47	|	0.32X |
|3k|	3.12 	|	7.38 	|	0.42X |
|4k|	3.77  	|	7.43  	|	0.50X |
|5k|	4.46 	|	7.47	|	0.59X |
|8k|	6.44 	|	7.49	|	0.85X |
|__10k__|	__7.82__ 	|	__7.69__	|	__1.01X__ |
|__20k__|	__14.54__	|   __7.80__	|   __1.86X__ |
|__50k__|	__34.35__ 	|	__8.31__ 	|	__4.13X__ |
|__80k__|	__54.80__ 	|	__8.68__ 	|	__6.31X__ |
|__100k__|	__68.82__ 	|	__9.07__ 	|	__7.58X__ |
|__110k__|  __75.92__	|   __8.99__	|   __8.43X__ |
|__120k__|  __83.03__	|   __9.52__	|   __8.71X__ |
|__150k__|  __104.24__	|   __9.92__	|   __10.50X__|
|__180k__|  __124.28__	|   __10.68__	|   __11.62X__|

Conclusion: Setting the threshold to 10K is good for __add operation__ of discontiguous tensors. 
   
Platform: Platinum 8180  
Operation: exp    
Tensor Continuity: discontiguous    
Unit: microsecond    

Time cost result is below:  

|Tensor Size|In series|In parallel|SpeedUp|
|---|---:|---:|---:|
|__1k__	|__10.02__	  |__7.27__	|	__1.37X__|
|__2k__	|__19.01__   	|__7.83__ |	__2.42X__|
|__3k__	|__27.73__   	|__7.48__ |	__3.70X__|
|__4k__	|__36.45__		|__7.66__	|	__4.75X__|
|__5k__	|__45.26__		|__8.13__	|	__5.56X__|
|__8k__	|__71.36__		|__8.70__	|	__8.19X__|
|__10k__|__88.75__		|__9.15__   |	__9.69X__|
|__20k__|__176.26__		|__11.32__  |	__15.56X__|
|__50k__|__439.68__		|__19.07__	|	__23.04X__|
|__80k__|__700.40__		|__26.99__	|	__25.94X__|
|__100k__|__876.42__	|__27.61__ 	|	__31.73X__|
|__110k__|__983.76__	|__29.79__	|	__33.01X__|
|__120k__|__1050.07__	|__31.87__	|	__32.94X__|
|__150k__|__1341.23__	|__37.59__	|	__35.67X__|
|__180k__|__1584.88__	|__43.27__	|	__36.62X__|  


Conclusion: Setting the threshold to 1K is good __exponential operation__ of contiguous tensors. 

Conclusions: 

- Discontiguous operation can be improved a lot by using OpenMP optimization.
- OpenMP threshold of discontiguous tensor is usually lower than that of contiguous tensor because the same operation of discontiguous tensor is more time-consuming than contiguous tensor. 


#### 4.3 LSTM benchmark test
To consolidate the performance boost benefiting from the elementwise optimization, we choose the a widely-used RNN unit: LSTM as the model-level benchmark reference. This is because:
1. LSTM related computations involve considerable elementwise operations;
2. PyTorch provides a scalable and flexible Python API to execute LSTM computation.

We retrieve the LSTM benchmark via the script:  https://github.com/xhzhao/pytorch-rnn-benchmark , and in which, 
1. The Python API torch.nn.LSTM is used as the entry of LSTM computation. 
2. We run the benchmarks on 24 selective input shapes utilized by different NLP models, 
3. The unit for benchmarks is Sentence Per Second (SPS). 
[N, T, D, Z] stands for batch size, embedding size, sentence length and hidden size.
Specifically, 
The [64, 50, 500, 500] is used by OpenNMT. The [64, 25, 4096, 4096] is used by Deepbench.

Platform: Platinum-8180  
Phase: Inference  
Unit: SPS(Scentence per Sencond)  

|LSTM	Input Shape|Xeon Platinum 8180 OOB|Xeon Platinum 8180 Optimized|SpeedUp|
|---|---:|---:|---:|
|[64, 15, 500, 500]|899.4494|7393.76|__8.22X__|
|[64, 20, 500, 500]|937.1688|5895.53|__6.29X__|
|[64, 25, 500,500]|750.8159|4808.17|__6.40X__|
|[64, 30, 500,500]|625.825|2351.56|__3.76X__|
|[64, 35, 500,500]|536.1393|3446.69|__6.43X__|
|[64, 40, 500,500]|469.1356|2907.74|__6.20X__|
|[64, 45, 500,500]|417.338|2502.57|__6.00X__|
|[64, 50, 500,500]|375.6814|2412.96|__6.43X__|
|[16, 25, 512, 512]|474.9601|1325.45|__2.79X__|
|[32, 25, 512, 512]|606.5853|2394.69|__3.95X__|
|[64, 25, 512, 512]|700.1314|3661.21|__5.23X__|
|[128, 25, 512, 512]|771.5298|4931.85|__6.39X__|
|[16, 25, 1024, 1024]|195.6518|434.34|__2.22X__|
|[32, 25, 1024, 1024]|261.1828|792.48|__3.03X__|
|[64, 25, 1024, 1024]|323.7316|1174.23|__3.62X__|
|[128, 25, 1024, 1024]|458.3642|1793.54|__3.91X__|
|[16, 25, 2048, 2048]|48.7229|71.07|__1.46X__|
|[32, 25, 2048, 2048]|77.4796|131.74|__1.70X__|
|[64, 25, 2048, 2048]|132.8328|245.78|__1.85X__|
|[128, 25, 2048, 2048]|178.2548|429.59|__2.41X__|
|[16, 25, 4096, 4096]|12.4995|16.99|__1.36X__|
|[32, 25, 4096, 4096]|23.0582|28.89|__1.25X__|
|[64, 25, 4096, 4096]|39.3725|53.48|__1.36X__|
|[128, 25, 4096, 4096]|61.866|97.97|__1.58X__|

Platform: Platinum-8180  
Phase: Training  
Unit: SPS(Scentence per Sencond)  

|LSTM	Input Shape|Xeon Platinum 8180 OOB|Xeon Platinum 8180 Optimized|Speed-up|
|---|---:|---:|---:|
|[64, 15, 500, 500] |432.5038|740.19|__1.71X__|
|[64, 20, 500, 500] |385.2532|506.49|__1.31X__|
|[64, 25, 500,500] |308.066|476.33|__1.55X__|
|[64, 30, 500,500]|264.2467|406.49|__1.54X__|
|[64, 35, 500,500]|217.2079|362.4|__1.67X__|
|[64, 40, 500,500]|199.5474|321.25|__1.61X__|
|[64, 45, 500,500]|187.0923|292.01|__1.56X__|
|[64, 50, 500,500]|159.5678|255.32|__1.60X__|
|[16, 25, 512, 512]|168.2578|269.11|__1.60X__|
|[32, 25, 512, 512]|217.3134|365.27|__1.68X__|
|[64, 25, 512, 512]|273.1848|475.26|__1.74X__|
|[128, 25, 512, 512]|320.5748|549.36|__1.71X__|
|[16, 25, 1024, 1024]|62.4692|89.46|__1.43X__|
|[32, 25, 1024, 1024]|89.6243|144.03|__1.61X__|
|[64, 25, 1024, 1024]|127.414|199.49|__1.57X__|
|[128, 25, 1024, 1024]|174.6576|255.07|__1.46X__|
|[16, 25, 2048, 2048]|18.8309|25.69|__1.36X__|
|[32, 25, 2048, 2048]|30.9957|47.01|__1.52X__|
|[64, 25, 2048, 2048]|51.2821|75.98|__1.48X__|
|[128, 25, 2048, 2048]|71.7206|113.27|__1.58X__|
|[16, 25, 4096, 4096]|6.0788|7.46|__1.23X__|
|[32, 25, 4096, 4096]|10.954|13.98|__1.28X__|
|[64, 25, 4096, 4096]|18.5955|24.85|__1.34X__|
|[128, 25, 4096, 4096]|28.1366|39.01|__1.39X__|


Platform: CPU E5-2699 v4  
Phase: Inference  
Unit: SPS(Scentence per Sencond)  

|LSTM	Input Shape|Xeon E5-2699 OOB|Xeon E5-2699 Optimized|Speed-up|
|---|---:|---:|---:|
|[64, 15, 500, 500]|1169.737|6135.84|__5.24X__|
|[64, 20, 500, 500]|923.5499|5367.74|__5.81X__|
|[64, 25, 500,500]|739.8101|1479.79|__2.00X__|
|[64, 30, 500,500]|618.0939|4087.51|__6.61X__|
|[64, 35, 500,500]|528.3323|3485.47|__6.60X__|
|[64, 40, 500,500]|462.2187|3037.45|__6.57X__|
|[64, 45, 500,500]|410.5386|2674.81|__6.51X__|
|[64, 50, 500,500]|369.9179|2419.29|__6.54X__|
|[16, 25, 512, 512]|639.4213|2179.69|__3.41X__|
|[32, 25, 512, 512]|680.3161|3589.74|__5.28X__|
|[64, 25, 512, 512]|727.8996|4917.65|__6.76X__|
|[128, 25, 512, 512]|760.9095|5852.11|__7.69X__|
|[16, 25, 1024, 1024]|320.0169|1368.51|__4.28X__|
|[32, 25, 1024, 1024]|349.7738|1926.68|__5.51X__|
|[64, 25, 1024, 1024]|368.3568|2293.78|__6.23X__|
|[128, 25, 1024, 1024]|490.1187|2523.87|__5.15X__|
|[16, 25, 2048, 2048]|137.989|379.72|__2.75X__|
|[32, 25, 2048, 2048]|159.1569|590.53|__3.71X__|
|[64, 25, 2048, 2048]|214.677|721|__3.36X__|
|[128, 25, 2048, 2048]|210.0029|685.11|__3.26X__|
|[16, 25, 4096, 4096]|42.7353|69.16|__1.62X__|
|[32, 25, 4096, 4096]|66.9777|129.84|__1.94X__|
|[64, 25, 4096, 4096]|82.5284|181.46|__2.20X__|
|[128, 25, 4096, 4096]|83.1054|179.11|__2.16X__|

Platform: CPU E5-2699 v4  
Phase: Training  
Unit: SPS(Scentence per Sencond)  

|LSTM	Input Shape|Xeon E5-2699 OOB|Xeon E5-2699 Optimized|Speed-up|
|---|---:|---:|---:|
|[64, 15, 500, 500]|451.2899|627.66|_|
|[64, 20, 500, 500]|370.242|497.26|_|
|[64, 25, 500,500]|298.1386|363.61|_|
|[64, 30, 500,500]|251.8914|327.72|_|
|[64, 35, 500,500]|225.749|285.99|_|
|[64, 40, 500,500]|192.7014|271.03|_|
|[64, 45, 500,500]|175.5287|245.5|_|
|[64, 50, 500,500]|161.343|229.74|_|
|[16, 25, 512, 512]|207.6788|201.7|_|
|[32, 25, 512, 512]|250.4016|301.76|_|
|[64, 25, 512, 512]|306.2745|429.34|_|
|[128, 25, 512, 512]|345.1608|456.06|_|
|[16, 25, 1024, 1024]|66.2632|67.93|_|
|[32, 25, 1024, 1024]|37.8289|114.71|_|
|[64, 25, 1024, 1024]|76.6716|173.85|_|
|[128, 25, 1024, 1024]|141.6185|218|_|
|[16, 25, 2048, 2048]|20.5789|20.82|_|
|[32, 25, 2048, 2048]|34.5047|36.93|_|
|[64, 25, 2048, 2048]|55.1509|62.73|_|
|[128, 25, 2048, 2048]|71.7717|88.76|_|
|[16, 25, 4096, 4096]|6.8679|7.09|_|
|[32, 25, 4096, 4096]|12.5718|13.85|_|
|[64, 25, 4096, 4096]|20.1554|23.66|_|
|[128, 25, 4096, 4096]|27.4074|34.49|_|


Conclusion: 
According to the benchmarks retrieved on Intel Xeon Platinum 8180, 
1. For LSTM inference (forward-only), the performance is get boosted from 1.25X to 8.22X.
2. For LSTM training (forward + backward), the performance is get boosted from 1.23X to 1.74X.

Test results analysis:
1. For inference benchmarks: As the contributions of elementwise operation varies from the different input shapes,  it is expected the performance boosts are not uniform with input shape changing. 
2. For training benchmarks: Apart from sharing the same reason of inference benchmarks. As the backward computation gains less from the elementwise optimization, it is expected the performance boosts on training benchmarks are not outstanding as inference benchmarks, and not uniform with input shape changing.
