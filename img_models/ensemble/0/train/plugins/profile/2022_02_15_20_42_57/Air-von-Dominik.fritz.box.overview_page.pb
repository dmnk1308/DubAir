?	?O0???@?O0???@!?O0???@	????I@????I@!????I@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?O0???@????*@A7??㗐@Yo????I@rEagerKernelExecute 0*	sh??<??@2
HIterator::Model::MaxIntraOpParallelism::MapAndBatch::FiniteTake::Shufflek?ܘ??:@!???j?D@)?ܘ??:@1???j?D@:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::MapAndBatchX9??v/@!iT?x?9@)X9??v/@1iT?x?9@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismR????9@!9|?&?D@)?V-$@1?Pm
0@:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::MapAndBatch::FiniteTake}Ǹ?⨬@@!l?C??J@){C"@1~/?)'@:Preprocessing2F
Iterator::Model??(\??=@!?[?-vG@)??Q??@1?S??6x@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9????I@I@???e?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????*@????*@!????*@      ?!       "      ?!       *      ?!       2	7??㗐@7??㗐@!7??㗐@:      ?!       B      ?!       J	o????I@o????I@!o????I@R      ?!       Z	o????I@o????I@!o????I@b      ?!       JCPU_ONLYY????I@b q@???e?W@Y      Y@q?HD	<???"?
device?Your program is NOT input-bound because only 4.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 