	???Bb@???Bb@!???Bb@	@?A
F"??@?A
F"??!@?A
F"??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:???Bb@?HZ?-@AK?*?a@Y???i????rEagerKernelExecute 0*	??? 0??@2?
HIterator::Model::MaxIntraOpParallelism::MapAndBatch::FiniteTake::Shuffle??d???!=??P?I@)?d???1=??P?I@:Preprocessing2w
?Iterator::Model::MaxIntraOpParallelism::MapAndBatch::FiniteTake?(
??<	??!???Bh?W@)C??fZ??1o??E@:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::MapAndBatchq=
ףp??!jEy?py	@)q=
ףp??1jEy?py	@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???S???!??6|??@)L7?A`???1b[??H??:Preprocessing2F
Iterator::Model?Os?"??!ޕ6?{@){/?h?r?1?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??A
F"??I?o}n7?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?HZ?-@?HZ?-@!?HZ?-@      ?!       "      ?!       *      ?!       2	K?*?a@K?*?a@!K?*?a@:      ?!       B      ?!       J	???i???????i????!???i????R      ?!       Z	???i???????i????!???i????b      ?!       JCPU_ONLYY??A
F"??b q?o}n7?X@