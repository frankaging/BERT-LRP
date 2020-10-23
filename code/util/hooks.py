def init_hooks_lrp(model):
    """
    Initialize all the hooks required for full lrp for BERT model.
    """
    # in order to backout all the lrp through layers
    # you need to register hooks here.
    model.bert.embeddings.register_forward_hook(
        get_activation('embeddings'))