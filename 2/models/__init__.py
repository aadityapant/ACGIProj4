def create_model(opt):
    from .mesh_classifier import ClassifierModel # todo - get rid of this ?
    model = ClassifierModel(opt)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    return model
