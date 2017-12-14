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
`OPERATION`—— operation, copyadd/div/sin/exp/sum/prod 

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

|LSTM	|Xeon Platinum 8180 OOB|Xeon Platinum 8180-Optimized|Speed-up|
|---|---:|---:|---:|
|__[64, 15, 500, 500]|__899.4494|__7393.76|__|
|__[64, 20, 500, 500]|__937.1688|__5895.53|__|
|__[64, 25, 500,500]|__750.8159|__4808.17|__|
|__[64, 30, 500,500]|__625.825|__2351.56|__|
|__[64, 35, 500,500]|__536.1393|__3446.69|__|
|__[64, 40, 500,500]|__469.1356|__2907.74|__|
|__[64, 45, 500,500]|__417.338|__2502.57|__|
|__[64, 50, 500,500]|__375.6814|__2412.96|__|
|__[16, 25, 512, 512]|__474.9601|__1325.45|__|
|__[32, 25, 512, 512]|__606.5853|__2394.69|__|
|__[64, 25, 512, 512]|__700.1314|__3661.21|__|
|__[128, 25, 512, 512]|__771.5298|__4931.85|__|
|__[16, 25, 1024, 1024]|__195.6518|__434.34|__|
|__[32, 25, 1024, 1024]|__261.1828|__792.48|__|
|__[64, 25, 1024, 1024]|__323.7316|__1174.23|__|
|__[128, 25, 1024, 1024]|__458.3642|__1793.54|__|
|__[16, 25, 2048, 2048]|__48.7229|__71.07|__|
|__[32, 25, 2048, 2048]|__77.4796|__131.74|__|
|__[64, 25, 2048, 2048]|__132.8328|__245.78|__|
|__[128, 25, 2048, 2048]|__178.2548|__429.59|__|
|__[16, 25, 4096, 4096]|__12.4995|__16.99|__|
|__[32, 25, 4096, 4096]|__23.0582|__28.89|__|
|__[64, 25, 4096, 4096]|__39.3725|__53.48|__|
|__[128, 25, 4096, 4096]|__61.866|__97.97|__|


Conclusion: 
According to the benchmarks retrieved on Intel Xeon Platinum 8180, 
1. For LSTM inference (forward-only), the performance is get boosted from xx to xx.
2. For LSTM training (forward + backward), the performance is get boosted from xx to xx.

We retrieve the LSTM benchmark via the script:  https://github.com/xhzhao/pytorch-rnn-benchmark , and in which, 
1. The Python API torch.nn.LSTM is used as the entry of LSTM computation. 
2. We run the benchmarks on 24 selective input shapes utilized by different NLP models, 
3. The unit for benchmarks is Sentence Per Second (SPS). 
[N, T, D, Z] stands for batch size, embedding size, sentence length and hidden size.
Specifically, 
The [64, 50, 500, 500] is used by OpenNMT. The [64, 25, 4096, 4096] is used by Deepbench.

Test results analysis:
1. For inference benchmarks: As the contributions of elementwise operation varies from the different input shapes,  it is expected the performance boosts are not uniform with input shape changing. 
2. For training benchmarks: Apart from sharing the same reason of inference benchmarks. As the backward computation gains less from the elementwise optimization, it is expected the performance boosts on training benchmarks are not outstanding as inference benchmarks, and not uniform with input shape changing.
