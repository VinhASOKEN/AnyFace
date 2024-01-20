Class_Info = {
    'num'  : 4,   
    'name' : [
        'mid-light',
        'light',
        'mid-dark',
        'dark'
    ]
}
Train_Config = {
    'path'            : "/data/disk2/vinhnguyen/AnyFace/data/train",
    'class'           : Class_Info,
    'image_size'      : (224, 224),
    'epoch'           : 38,
    'batch_size'      : 112,
    'learning_rate'   : 1e-5,
    'model_save_path' : '/data/disk2/vinhnguyen/AnyFace/model_skintone/weights',
    'load_checkpoint' : None
}

Valid_Config = {
    'path'            : "/data/disk2/vinhnguyen/AnyFace/data/valid",
    'class'           : Class_Info,
    'image_size'      : (224, 224),
    'batch_size'      : 112
}

Testing_Config = {
    'class'           : Class_Info,
    'image_size'      : (224, 224),
    'load_checkpoint' : '/data/disk2/vinhnguyen/AnyFace/model_skintone/weights/best.pth'
}