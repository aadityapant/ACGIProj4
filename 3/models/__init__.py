def create_model(opt):
    from .mesh_classifier import ClassifierModel # todo - get rid of this ?
    model = ClassifierModel(opt)
    total_params, trainable_params = model.get_parameter_count()
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    return model
