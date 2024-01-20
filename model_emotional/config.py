Class_Info = {
    'num'  : 7,   
    'name' : [
        'Neutral',
        'Happiness',
        'Anger',
        'Surprise',
        'Fear',
        'Sadness',
        'Disgust'
    ]
}
Train_Config = {
    'path'            : "/data/disk2/vinhnguyen/AnyFace/data/train",
    'class'           : Class_Info,
    'image_size'      : (224, 224),
    'epoch'           : 38,
    'batch_size'      : 150,
    'learning_rate'   : 1e-5,
    'model_save_path' : '/data/disk2/vinhnguyen/AnyFace/model_emotional/weights',
    'load_checkpoint' : None
}

Valid_Config = {
    'path'            : "/data/disk2/vinhnguyen/AnyFace/data/valid",
    'class'           : Class_Info,
    'image_size'      : (224, 224),
    'batch_size'      : 150
}

Testing_Config = {
    'class'           : Class_Info,
    'image_size'      : (224, 224),
    'load_checkpoint' : '/data/disk2/vinhnguyen/AnyFace/model_emotional/weights/best.pth'
}