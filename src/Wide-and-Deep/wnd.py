import time
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# print('Using TensorFlow version: ', tf.__version__)

train_path = '../../data/train.csv'
test_path = '../../data/test.csv'

# field = ['hour', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain', 'slotid', 'slotwidth',
#          'slotheight', 'slotvisibility', 'slotformat', 'creative', 'keypage', 'advertiser', 'usertag']

field = ['weekday', 'hour', 'bidid', 'timestamp', 'logtype', 'ipinyouid', 'useragent', 'IP', 'region', 'city',
         'adexchange', 'domain', 'url', 'urlid', 'slotid',
         'slotwidth', 'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'creative', 'bidprice', 'payprice',
         'keypage', 'advertiser', 'usertag']

CATEGORICAL_COLUMNS = field
LABEL_COLUMN = ['click']

TRAIN_DATA_COLUMNS = LABEL_COLUMN + CATEGORICAL_COLUMNS

FEATURE_COLUMN = CATEGORICAL_COLUMNS

print(FEATURE_COLUMN)

batch_size = 400


def generate_input_fn(filename, batch_size=batch_size):
    def _input_fn():
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TextLineReader(skip_header_lines=1)
        # Reads out batch_size number of lines
        key, value = reader.read_up_to(filename_queue, num_records=batch_size)
        print(key, value)

        cate_defaults = [[' '] for x in field]
        label_defaults = [[0]]

        column_headers = TRAIN_DATA_COLUMNS

        record_defaults = label_defaults + cate_defaults

        columns = tf.decode_csv(value, record_defaults=record_defaults)

        all_columns = dict(zip(column_headers, columns))

        labels = all_columns.pop(LABEL_COLUMN[0])

        features = all_columns
        for feature_name in CATEGORICAL_COLUMNS:
            features[feature_name] = tf.expand_dims(features[feature_name], -1)
        return features, labels

    return _input_fn


wide_columns = []
for name in CATEGORICAL_COLUMNS:
    wide_columns.append(tf.contrib.layers.sparse_column_with_hash_bucket(name, hash_bucket_size=1000))

deep_columns = []
# for name in CATEGORICAL_COLUMNS:
#     deep_columns.append(tf.contrib.layers.sparse_column_with_hash_bucket(name, hash_bucket_size=1000))
# deep_columns.append(tf.contrib.layers.embedding_column(name, dimension=8))

# Embedding for wide columns into deep columns
for col in wide_columns:
    deep_columns.append(tf.contrib.layers.embedding_column(col, dimension=8))
print(wide_columns)
print('deep_columns:', deep_columns)


def create_model_dir(model_type):
    return 'model/model_' + model_type + '_' + str(int(time.time()))


def get_model(model_type, model_dir):
    print('Model directory = {0}'.format(model_dir))

    runconfig = tf.contrib.learn.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=100)

    m = None

    if model_type == 'WIDE':
        m = tf.contrib.learn.LogisticRegressor(model_dir=model_dir, feature_columns=wide_columns)

    if model_type == 'DEEP':
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir, feature_columns=wide_columns, hide_units=[100, 50, 25])

    if model_type == 'WIDE_AND_DEEP':
        m = tf.contrib.learn.DNNLinearCombinedRegressor(model_dir=model_dir,
                                                        linear_feature_columns=wide_columns,
                                                        dnn_feature_columns=deep_columns,
                                                        dnn_hidden_units=[100, 70, 50, 25],
                                                        config=runconfig)
    print('estimator built')

    return m


MODEL_TYPE = 'WIDE_AND_DEEP'
model_dir = create_model_dir(model_type=MODEL_TYPE)
m = get_model(model_type=MODEL_TYPE, model_dir=model_dir)

from tensorflow.contrib.learn.python.learn import evaluable

print(isinstance(m, evaluable.Evaluable))

train_sample_size = 3000000
train_steps = train_sample_size / batch_size

for round in range(1):
    m.fit(input_fn=generate_input_fn(train_path, batch_size), steps=train_steps)
print('fit done')

# Evaluate
eval_sample_size = 300000
eval_steps = eval_sample_size / batch_size

result = m.evaluate(input_fn=generate_input_fn(test_path), steps=eval_steps)

print('evaluate done')
# print('Accuracy:{0}'.format(result['accuracy']))
print(result)

pred = m.predict(input_fn=generate_input_fn(test_path))
print(pred)


def pred_fn():
    sample = ['5','00','623e5066cada03264c96de094d8241f9','20130612000103148','1','Vh5zZAnROUSCgma','windows_ie','222.178.10.*','275','275','2','lAB-VDz4LpscFsf',
'a6c88104e0a64c3325fba39fd5b87702','null','4210897124','250','250','2','0','5','2abc9eaf57d17a96195af3f63c45dc72','300','17','bebefa5efe83beee17a3d245e7c5085b','1458','10057']

    # fi = open(test_path, 'r')
    # next(fi)
    # for line in fi:
    #     s =

    sample_dict = dict(zip(FEATURE_COLUMN, sample))
    print(sample_dict)

    for feature_name in CATEGORICAL_COLUMNS:
        sample_dict[feature_name] = tf.expand_dims(sample_dict[feature_name], -1)

    # for feature_name in CONTINUOUS_COLUMN:
    #     sample_dict[feature_name] = tf.constant(sample_dict[feature_name], dtype=tf.int32)
    print(sample_dict)

    return sample_dict


pred = m.predict_scores(input_fn=pred_fn)
print(pred)
