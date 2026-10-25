def parameter_summary(module):
    total=sum(p.numel() for p in module.parameters())
    trainable=sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {'parameters':total,'trainable_parameters':trainable}
