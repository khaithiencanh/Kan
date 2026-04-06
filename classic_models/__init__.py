from .vision_kansformer import kit_base_patch16_224, kit_base_patch16_224_in21k

cfgs = {
    'kansformer1': kit_base_patch16_224,
    'kansformer2': kit_base_patch16_224_in21k,
}

def find_model_using_name(model_name, num_classes):   
    # Hàm này sẽ trả về mô hình Kansformer dựa trên tên bạn truyền vào
    if model_name in cfgs:
        return cfgs[model_name](num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} không tìm thấy. Hãy chọn kansformer1 hoặc kansformer2")