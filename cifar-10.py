import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
import datetime
import os

# 1. 데이터 로드 및 tf.data 파이프라인 구성 (속도 향상의 핵심)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

y_train_oh = tf.one_hot(y_train.flatten(), 10)
y_test_oh = tf.one_hot(y_test.flatten(), 10)

BATCH_SIZE = 128
WEIGHT_DECAY = 1e-3
EPOCHS = 100

# 데이터 증강 레이어
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

# 학습용 데이터셋 파이프라인
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_oh))
train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE)
# CPU가 미리 증강과 배치를 준비하게 함 (Prefetch)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# 검증용 데이터셋 파이프라인
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test_oh)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 2. Residual Block (L2 규제 포함)
def residual_block(x, filters, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding='same', 
                      kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, 3, padding='same',
                      kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    x = layers.BatchNormalization()(x)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                 kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

# 3. 모델 빌드 (채널 수 최적화: 64-128-256)
def build_resnet():
    inputs = layers.Input(shape=(32, 32, 3))
    
    # 초기 레이어 (64채널)
    x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual Stages (너비 축소로 연산량 절감)
    x = residual_block(x, 64, stride=1)
    x = layers.Dropout(0.2)(x)
    
    x = residual_block(x, 256, stride=2) # 16x16
    x = layers.Dropout(0.3)(x)
    
    x = residual_block(x, 512, stride=2) # 8x8
    x = layers.Dropout(0.4)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    
    return models.Model(inputs, outputs=x)

model = build_resnet()

# 4. 스케줄러 및 컴파일
cosine_lr = optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=EPOCHS * (len(x_train) // BATCH_SIZE)
)

model.compile(
    optimizer=optimizers.AdamW(learning_rate=cosine_lr, weight_decay=WEIGHT_DECAY),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# 5. 실행
model.fit(
    train_ds,  # 파이프라인 사용
    epochs=EPOCHS,
    validation_data=test_ds,
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir="logs/fast_resnet")]
)