from transformers import AutoTokenizer

# ============ Inisialisasi Model IndoBERT ============
tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")

def prepare_input(riwayat_tuntutan, fakta, fakta_hukum, pertimbangan_hukum):
    # Gabungkan semua input
    combined_text = (
        str(riwayat_tuntutan) + " [SEP] " +
        str(fakta) + " [SEP] " +
        str(fakta_hukum) + " [SEP] " +
        str(pertimbangan_hukum)
    )

    # Tokenisasi
    inputs = tokenizer(
        combined_text,
        padding="max_length",   # biar panjang sama
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    return inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]