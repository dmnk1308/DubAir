	5^?I?{@5^?I?{@!5^?I?{@	M??]???M??]???!M??]???"}
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails<	5^?I?{@?E???T@A+????z@Y?|?5^:@rEagerKernelExecute 9*	??n??@2?
HIterator::Model::MaxIntraOpParallelism::MapAndBatch::FiniteTake::Shuffle????k?&@!?&?=q?Q@)???k?&@1?&?=q?Q@:Preprocessing2w
?Iterator::Model::MaxIntraOpParallelism::MapAndBatch::FiniteTake??~?^??.@!?????X@)??۟?F@1??V? <@:Preprocessing2F
Iterator::ModelˡE?????!
??Qm??)ˡE?????1
??Qm??:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::MapAndBatch????????!ǚi???)????????1ǚi???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9M??]???I?<???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?E???T@?E???T@!?E???T@      ?!       "      ?!       *      ?!       2	+????z@+????z@!+????z@:      ?!       B      ?!       J	?|?5^:@?|?5^:@!?|?5^:@R      ?!       Z	?|?5^:@?|?5^:@!?|?5^:@b      ?!       JCPU_ONLYYM??]???b q?<???X@