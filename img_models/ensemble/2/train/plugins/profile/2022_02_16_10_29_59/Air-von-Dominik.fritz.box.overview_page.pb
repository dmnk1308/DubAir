?	??.]i@??.]i@!??.]i@	5?c????5?c????!5?c????"}
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails<??.]i@w?*2:l<@A??d??e@Y?? ?	??rEagerKernelExecute 2*	R??k?@2?
HIterator::Model::MaxIntraOpParallelism::MapAndBatch::FiniteTake::Shuffle?c}????!5????;K@)c}????15????;K@:Preprocessing2w
?Iterator::Model::MaxIntraOpParallelism::MapAndBatch::FiniteTake????? @!l??~?X@)?y????1?È<x?E@:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::MapAndBatch? ??	L??!??|c????)? ??	L??1??|c????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism`!sePm??!d????)kD0.??1???%??:Preprocessing2F
Iterator::Model???w?-??!??U`???)j?t?v?1??Ov???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 14.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no95?c????I
N??#?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	w?*2:l<@w?*2:l<@!w?*2:l<@      ?!       "      ?!       *      ?!       2	??d??e@??d??e@!??d??e@:      ?!       B      ?!       J	?? ?	???? ?	??!?? ?	??R      ?!       Z	?? ?	???? ?	??!?? ?	??b      ?!       JCPU_ONLYY5?c????b q
N??#?X@Y      Y@q9L?0O??"?
both?Your program is POTENTIALLY input-bound because 14.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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