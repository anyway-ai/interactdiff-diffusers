import torch


def prepare_interactdiff_inputs(models, batch, batch_size, device):
    max_objs = 30

    """interactdiffusion_subject_phrases = interactdiffusion_subject_phrases[:max_objs]
    interactdiffusion_subject_boxes = interactdiffusion_subject_boxes[:max_objs]
    interactdiffusion_object_phrases = interactdiffusion_object_phrases[:max_objs]
    interactdiffusion_object_boxes = interactdiffusion_object_boxes[:max_objs]
    interactdiffusion_action_phrases = interactdiffusion_action_phrases[:max_objs]

    # prepare batched input to the InteractDiffusionInteractionProjection (boxes, phrases, mask)
    # Get tokens for phrases from pre-trained CLIPTokenizer
    tokenizer_inputs = models['tokenizer'](interactdiffusion_subject_phrases+interactdiffusion_object_phrases+interactdiffusion_action_phrases,
                                        padding=True, return_tensors="pt").to(device)
    
    # For the token, we use the same pre-trained text encoder
    # to obtain its text feature
    _text_embeddings = models['text_encoder'](**tokenizer_inputs).pooler_output
    n_objs = min(len(interactdiffusion_subject_boxes), max_objs)


    # For each entity, described in phrases, is denoted with a bounding box,
    # we represent the location information as (xmin,ymin,xmax,ymax)
    encoder_dtype = models['text_encoder'].dtype

    subject_boxes = torch.zeros(max_objs, 4, device=device, dtype=encoder_dtype)
    object_boxes = torch.zeros(max_objs, 4, device=device, dtype=encoder_dtype)

    subject_boxes[:n_objs] = torch.tensor(interactdiffusion_subject_boxes[:n_objs])
    object_boxes[:n_objs] = torch.tensor(interactdiffusion_object_boxes[:n_objs])
    
    subject_text_embeddings = torch.zeros(max_objs, 768, device=device, dtype=encoder_dtype)
    subject_text_embeddings[:n_objs] = _text_embeddings[:n_objs*1]
    
    object_text_embeddings = torch.zeros(max_objs, 768, device=device, dtype=encoder_dtype)
    object_text_embeddings[:n_objs] = _text_embeddings[n_objs*1:n_objs*2]
    
    action_text_embeddings = torch.zeros(max_objs, 768, device=device, dtype=encoder_dtype)
    action_text_embeddings[:n_objs] = _text_embeddings[n_objs*2:n_objs*3]"""

    encoder_dtype = models.dtype

    subject_boxes = batch['subject_boxes']
    object_boxes = batch['object_boxes']
    masks = batch['masks']
    subject_text_embeddings = batch["subject_text_embeddings"]
    object_text_embeddings = batch["object_text_embeddings"]
    action_text_embeddings = batch["action_text_embeddings"]
    
    # Generate a mask for each object that is entity described by phrases
    masks = torch.zeros(max_objs, device=device, dtype=encoder_dtype)
    n_objs = min(len(subject_boxes), max_objs)
    masks[:n_objs] = 1

    repeat_batch = batch_size
    subject_boxes = subject_boxes.expand(repeat_batch, -1, -1).clone()
    object_boxes = object_boxes.expand(repeat_batch, -1, -1).clone()
    subject_text_embeddings = subject_text_embeddings.expand(repeat_batch, -1, -1).clone()
    object_text_embeddings = object_text_embeddings.expand(repeat_batch, -1, -1).clone()
    action_text_embeddings = action_text_embeddings.expand(repeat_batch, -1, -1).clone()
    masks = masks.expand(repeat_batch, -1).clone()

    cross_attention_kwargs = {}
    cross_attention_kwargs['gligen'] = {
            'subject_boxes': subject_boxes,
            'object_boxes': object_boxes,
            'subject_positive_embeddings': subject_text_embeddings,
            'object_positive_embeddings': object_text_embeddings,
            'action_positive_embeddings': action_text_embeddings,
            'masks': masks
        }
    
    return cross_attention_kwargs



class HOIGroundingNetInput:
    def __init__(self):
        self.set = False

    def prepare(self, batch):
        """
        batch should be the output from dataset.
        Please define here how to process the batch and prepare the
        input only for the ground tokenizer.
        batch = {
            'subject_boxes': [[..],[..],..]],
            'object_boxes': [[..],[..],..]],
            'masks': ..,
            'subject_text_embeddings': [..],
            'object_text_embeddings': [..],
            'action_text_embeddings': [..]
        }
        """

        self.set = True

        subject_boxes = batch['subject_boxes']
        object_boxes = batch['object_boxes']
        masks = batch['masks']
        subject_positive_embeddings = batch["subject_text_embeddings"]
        object_positive_embeddings = batch["object_text_embeddings"]
        action_positive_embeddings = batch["action_text_embeddings"]
        # batch["image_embeddings"]

        self.batch, self.max_box, self.in_dim = subject_positive_embeddings.shape
        self.device = subject_positive_embeddings.device
        self.dtype = subject_positive_embeddings.dtype

        return {"subject_boxes": subject_boxes, "object_boxes": object_boxes,
                "masks": masks,
                "subject_positive_embeddings": subject_positive_embeddings,
                "object_positive_embeddings": object_positive_embeddings,
                "action_positive_embeddings": action_positive_embeddings}

    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        Guidance for training (drop) or inference,
        please define the null input for the grounding tokenizer
        """

        assert self.set, "not set yet, cannot call this funcion"
        batch = self.batch if batch is None else batch
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype

        subject_boxes = object_boxes = torch.zeros(batch, self.max_box, 4, ).type(dtype).to(device)
        masks = torch.zeros(batch, self.max_box).type(dtype).to(device)
        subject_positive_embeddings = object_positive_embeddings = action_positive_embeddings \
            = torch.zeros(batch, self.max_box, self.in_dim).type(dtype).to(device)

        return {"subject_boxes": subject_boxes, "object_boxes": object_boxes,
                "masks": masks,
                "subject_positive_embeddings": subject_positive_embeddings,
                "object_positive_embeddings": object_positive_embeddings,
                "action_positive_embeddings": action_positive_embeddings}


def preprocessor(batch):
    grounding_tokenizer_input = HOIGroundingNetInput()
    return grounding_tokenizer_input.prepare(batch)