	??s?a?c@??s?a?c@!??s?a?c@	?c??H????c??H???!?c??H???"~
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails=??s?a?c@s	??@Aۋh;??b@Y:??KTo??rEagerKernelExecute 11*	Zd;???@2?
HIterator::Model::MaxIntraOpParallelism::MapAndBatch::FiniteTake::Shuffle?/5B?S???!???9?R@)/5B?S???1???9?R@:Preprocessing2w
?Iterator::Model::MaxIntraOpParallelism::MapAndBatch::FiniteTake???t?(?@!??H?JUX@)gaO;?5??1K?wD?5@:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::MapAndBatch?<֌r??!N??V???)?<֌r??1N??V???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismh?>????!"?A?;W@)?ʡE????1??ޯA???:Preprocessing2F
Iterator::Modeln?+????!?B?V?V@)?MbX9??1K?K?c???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?c??H???IΩ?[7?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	s	??@s	??@!s	??@      ?!       "      ?!       *      ?!       2	ۋh;??b@ۋh;??b@!ۋh;??b@:      ?!       B      ?!       J	:??KTo??:??KTo??!:??KTo??R      ?!       Z	:??KTo??:??KTo??!:??KTo??b      ?!       JCPU_ONLYY?c??H???b qΩ?[7?X@