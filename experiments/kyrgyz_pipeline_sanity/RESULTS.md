# Kyrgyz Pipeline Sanity

## Execution Record

- Baseline command: `/opt/venv/bin/python -c "import peft; print(peft.__version__)"`; result: `0.18.1`.
- Initial parity command: `ssh -p 31101 root@36.150.116.206 "timeout 1200 /opt/venv/bin/python /workspace/GLT/experiments/kyrgyz_pipeline_sanity/run_sanity.py"`; failure: `RuntimeError: cache parity mismatch; see cache_parity.json`.
- Successful command: `ssh -p 31101 root@36.150.116.206 "SANITY_CONTINUE_AFTER_CACHE_MISMATCH=1 timeout 720 /opt/venv/bin/python /workspace/GLT/experiments/kyrgyz_pipeline_sanity/run_sanity.py"`.
- Cache parity result: `all_exactly_identical=False`; within-mode reruns were stable, but cross-mode outputs diverged at frozen-base prompt 02 token 6 and LoRA English prompt 02 token 22.
- The micro-overfit generation uses `use_cache=False`, the full-sequence path, after the reproducible cache-path mismatch.
- The adapter tensor is outside Git-tracked files at `micro_overfit_lora/`; hashes are in `micro_overfit_metrics.json`.
- Cache parity wall time: `31.120s`; micro-overfit wall time: `78.723s`; final clean-stop check at `2026-08-05 23:44:24 UTC` (`07:44:24 Asia/Taipei`) showed baseline VRAM usage.

## Cache Parity

See `cache_parity.json` for all six prompts under both cache modes.

## Micro-Overfit Metrics

```json
{
  "model": "/workspace/model/real_SmolLM3-3B",
  "train_bin": "/workspace/kyrgyz_train.bin",
  "train_bin_sha256": "c63f76b1f870525c651f2743df54b0b6f75f7e26d02fc359219a88f309c03d72",
  "selected_data_indices": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15
  ],
  "selection_rule": "first 16 training sequences with no EOS, valid decode, and >=50% Cyrillic letters in the isolated first 192 tokens",
  "sequence_geometry": {
    "prompt_tokens": 64,
    "continuation_tokens": 128,
    "input_tokens": 192
  },
  "seed": 42,
  "steps": 500,
  "batch": 1,
  "learning_rate": 0.0002,
  "weight_decay": 0.01,
  "target_modules": [
    "gate_proj",
    "up_proj",
    "down_proj"
  ],
  "rank": 32,
  "alpha": 64,
  "dropout": 0.0,
  "bias": "none",
  "optimizer": "AdamW",
  "autocast_dtype": "torch.bfloat16",
  "trainable_parameters": 45121536,
  "loss_at_steps": {
    "1": 4.319334506988525,
    "100": 0.38742396235466003,
    "250": 0.02950780652463436,
    "500": 0.0015166960656642914
  },
  "final_training_loss": 0.0015166960656642914,
  "training_wall_time_s": 78.72301860013977,
  "peak_vram_allocated_mb": 7802.27587890625,
  "peak_vram_reserved_mb": 8068.0,
  "continuation_eval": {
    "ce": 0.014994445547927171,
    "ppl": 1.0151074262345121,
    "supervised_tokens": 2048
  },
  "generation": {
    "records": 16,
    "max_new_tokens": 128,
    "use_cache": false,
    "exact_token_match_rate": 0.88916015625,
    "mean_prefix_token_match_length": 113.5,
    "eos_rate": 0.0,
    "mean_repetition_rate": 0.46337890625,
    "invalid_decode_count": 1
  },
  "adapter_dir": "/workspace/GLT/experiments/kyrgyz_pipeline_sanity/micro_overfit_lora",
  "adapter_files": {
    "README.md": "4e72ac623b80ff3f7a10bdff4fc2a47f65127c899e120f733835b1ec5e413b4d",
    "adapter_config.json": "c065ade98b5018ffb2d3eb5b99945a09cde709dc431a9e9fb2205582f5312710",
    "adapter_model.safetensors": "abdd16d9c748efda98885b225855e484b7bdc3bc65ebb46ea9c12dcc9f015a1a"
  },
  "adapter_sha256_manifest": "108b5b1b6d798e51feb8cdb4cc73c9a231dffe996898edfcc1665c978969c118"
}
```

## All Prompt/Reference/Generated Triples

### 1. Data index `0`

**Prompt:** АКИpress'тин кыргыз тилиндеги маалыматтарынын толук архиви (коомдук-саясий жаңылыктар, маданият, кылмыш жана кырсыктар, сп

**Reference:** орт, ошондой эле аймактардагы маалыматтар) бир жаңылык950 сомСатып алуу бир жыл3500 сомкатталуу? Бул тариф жеке тараптарга гана тиешелүү. Эгерде материалдарга корпоративдик негизде каттала турган болсоңуз (уюмдар, мамлекеттик мекемелер, ведомстволор жана башка) биз

**Generated:** орт, ошондой эле аймактардагы маалыматтар) бир жаңылык950 сомСатып алуу бир жыл3500 сомкатталуу? Бул тариф жеке тараптарга гана тиешелүү. Эгерде материалдарга корпоративдик негизде каттала турган болсоңуз (уюмдар, мамлекеттик мекемелер, ведомстволор жана башка) биз

Metadata: exact_match=1.000000, prefix=128, length=128, termination=max_new_tokens, repetition=0.273438, invalid_decode=False

### 2. Data index `1`

**Prompt:** АГЫ 1916-ЖЫЛ 1916-ЖЫЛДАН КИЙИН 1916-ЖЫЛДАН КИЙИН: ЖЕКЕ ИЗИЛДӨӨЛӨР 1916-ЖЫЛД

**Reference:** АН КИЙИН: ТАРЫХ ТИЗМЕГИ 1916-ЖЫЛДАН КИЙИН: ДОКУМЕНТТЕР Адамдар ЭЛ АГАРТУУ ЖАНА АГАРТУУЧУЛАР Кыргызстандын инсандары. ЖУСУП Кыргызстандын инсандары. Лиля Турусбекова Кыргызстандын инсандары. Санжарбек Д

**Generated:** АН КИЙИН: ТАРЫХ ТИЗМЕГИ 1916-ЖЫЛДАН КИЙИН: ДОКУМЕНТТЕР Адамдар ЭЛ АГАРТУУ ЖАНА АГАРТУУЧУЛАР Кыргызстандын инсандары. ЖУСУП Кыргызстандын инсандары. Лиля Турусбекова Кыргызстандын инсандары. Санжарбек Д

Metadata: exact_match=1.000000, prefix=128, length=128, termination=max_new_tokens, repetition=0.546875, invalid_decode=False

### 3. Data index `2`

**Prompt:**  жана тарбиялоо жөнүндө МЕДИКТЕР ФОНДДУН ЖАШ ДАРЫГЕРГЕ СЫЙЛЫГЫ МЕДИЦИНАЛЫК ЖОЖдор ЖАНА

**Reference:**  БИЗДИН ФОНД ЖӨНҮНДӨ ФОНД ЖӨНҮНДӨ ФОНДДУН КАБАРЛАРЫ ДОЛБООРЛОРУБУЗ ЖАНА БИЗ ЖӨНҮНДӨ КЫЗМАТТАРЫБЫЗ, ЖУМШАЛГАН ЧЫГЫМДАР, ДОСТОРДУН КОЛДООСУ Башкы › Башкы бет ирети

**Generated:**  БИЗДИН ФОНД ЖӨНҮНДӨ ФОНД ЖӨНҮНДӨ ФОНДДУН КАБАРЛАРЫ ДОЛБООРЛОРУБУЗ ЖАНА БИЗ ЖӨНҮНДӨ КЫЗМАТТАРЫБЫЗ, ЖУМШАЛГАН ЧЫГЫМДАР, ДОСТОРДУН КОЛДООСУ Башкы › Башкы бет ирети

Metadata: exact_match=1.000000, prefix=128, length=128, termination=max_new_tokens, repetition=0.687500, invalid_decode=False

### 4. Data index `3`

**Prompt:**  ошентип БИРИНЧИ МУГАЛИМ сынагынын лауреаттарынын окуучулары менен 2022-жылдагы саякаты аяктады! Бул жолу август а

**Reference:** йында 2022-жылдын 23-апрелинде аяктаган 10-сынактын үч лауреатынын жана мурдагы жылдагы 9-сынактын бир лауреаты окуучулары менен саякатка келишти. Бизге төрт класс – Ат-Башы, Сузак, Ак-Суу жана Бакай-Ата райондорунан экинчи жана үчү

**Generated:** йында 2022-жылдын 23-апрелинде аяктаган 10-сынактын үч лауреатынын жана мурдагы жылдагы 9-сынактын бир лауреаты окуучулары менен саякатка келишти. Бизге төрт класс – Ат-Башы, Сузак, Ак-Суу жана Бакай-Ата райондорунан экинчи жана үчү

Metadata: exact_match=1.000000, prefix=128, length=128, termination=max_new_tokens, repetition=0.343750, invalid_decode=False

### 5. Data index `4`

**Prompt:** өкүлү, баардыгы – 287 адам! Айтмакчы, Бишкекке уюштурулган балдардын үч экскурсиясынан сырткары, ушул

**Reference:**  жайда мен фонддун атынан 8,-9 жана -10-сынакта БИРИНЧИ МУГАЛИМ аталган үч жеңүүчү мугалимдин (жана ошондой эле КММАда Илим күнүнүн алкагыда 2022-жылы өткөрүлгөн жаш окумуштуу дарыгерлер сынагы

**Generated:**  жайда мен фонддун атынан 8,-9 жана -10-сынакта БИРИНЧИ МУГАЛИМ аталган үч жеңүүчү мугалимдин (жана ошондой эле КММАда Илим күнүнүн алкагыда 2022-жылы өткөрүлгөн жаш окумуштуу дарыгерлер сынагы

Metadata: exact_match=1.000000, prefix=128, length=128, termination=max_new_tokens, repetition=0.367188, invalid_decode=False

### 6. Data index `5`

**Prompt:** ыдырышты. 3-класстын 29 окуучусу менен 10-сынактын жеңүүчүсү— Убайдылдаева Асель Уркалыковна Ысык-Кө

**Reference:** л облусунун Ак-Суу районундагы Чолпон айылындагы Утур Калиев атындагы мектептин мугалими. 3-класстын 32 окуучусу менен 10-сынактын жеңүүчүсү — Турганбаева Гүлүмкан Абыталыповна, Нарын облусунун Ат-Башы районунун Ача-Кайыңды айылынд

**Generated:** л облусунун Ак-Суу районундагы Чолпон айылындагы Утур Калиев атындагы мектептин мугалими. 3-класстын 32 окуучусу менен 10-сынактын жеңүүчүсү — Турганбаева Гүлүмкан Абыталыповна, Нарын облусунун Ат-Башы районунун Ача-Кайыңды айылынд

Metadata: exact_match=1.000000, prefix=128, length=128, termination=max_new_tokens, repetition=0.398438, invalid_decode=False

### 7. Data index `6`

**Prompt:** амдашып келген ата-энелер мурда борбор шаарга туугандарынын тойлоруна гана келишкендиктен, музей-театрларга кирмек тургай, ушундай теат

**Reference:** рлар, сейил бактардын бар экендигин да билишпегендигин мойнуна алышты. Сөздүн кыскасы, баары бул экскурсияны көптөн бери зарыга күтүп жүрүшкөнү айкын болду. Баса, мугалимдер экскурсия башталгандан аягына чейин видео тасмага т

**Generated:** рлар, сейил бактардын бар экендигин да билишпегендигин мойнуна алышты. Сөздүн кыскасы, баары бул экскурсияны көптөн бери зарыга күтүп жүрүшкөнү айкын болду. Баса, мугалимдер экскурсия башталгандан аягына чейин видео тасмага т

Metadata: exact_match=1.000000, prefix=128, length=128, termination=max_new_tokens, repetition=0.414062, invalid_decode=False

### 8. Data index `7`

**Prompt:** ) менен шаарда сейилдеп жүргөндө аябай ыңгайлуу экен, анткени балдарды алыстан эле айырмалап таанууга болот. Биз

**Reference:**  бир нече ыктыярчылардын жардамы менен балдардын баарына көз салып туруу үчүн, мугалимдер жана коштоп келген ата-энелер менен эки-экиден жетелешкен балдарды улам-улам санап жаттык. Окшош футболка кийген балдарды көзөмөлдөө бизге же

**Generated:**  бир нече ыктыярчылардын жардамы менен шаарда сейилдеп келген ата-энелер менен балдардын баарына көз салып туруу үчүн, бул жолу мурдагы балдардын баарына көз салып туруу үчүн, бул жолу мурдагы эки-экиден жетелешкен балдарды у

Metadata: exact_match=0.179688, prefix=21, length=128, termination=max_new_tokens, repetition=0.500000, invalid_decode=False

### 9. Data index `8`

**Prompt:** ты ичкен соң, Бишкекке алыстан жолдо чарчап, эс алышат, эртеси эртең менен уйкусу канган балдар күнү кечке шаар кы

**Reference:** дырышат. Кечинде суй жыгылып чарчаган балдар уктап алып, эртеси түшкө чейин дагы кызыктуу жайларды кыдырышып, түштөнүп алгандан кийин, үйлөрүнө узап кетишет. Бул жолу биз алгачкы жолу балдарды борбор калаабызд

**Generated:** дырышат. Кечинде суй жыгылып чарчаган балдар уктап алып, эртеси түшкө чейин дагы кызыктуу жайларды кыдырышып, түштөнүп алгандан кийин, бул жолу балдарды кыдырышып, бул жолу балдарды кыдырышып. Сөздүн кыскасы, бу

Metadata: exact_match=0.648438, prefix=83, length=128, termination=max_new_tokens, repetition=0.539062, invalid_decode=False

### 10. Data index `9`

**Prompt:**  окурмандарынын балдары бул аталган акылдуу спектаклге али бара элек болсо, анда бир барып көрүп коюуңуздарга кеңеш беребиз

**Reference:** . Балдарга жана ата-энелерге сөзсүз жагат! Биз ошондой эле Абдылас Малдыбаев атындагы Опера жана балет театрына бардык. Ал театр сезону аяктаганына байланыштуу августтан сентябрь айына чейин жабык болгондуктан, бул жолу мурдагы экскурсияга келишкен балдар көрг

**Generated:** . Балдарга жана ата-энелерге сөзсүз жагат! Биз ошондой эле Абдылас Малдыбаев атындагы Опера жана балет театрына бардык. Ал театр сезону аяктаганына байланыштуу августтан сентябрь айына чейин жабык болду, бул жолу мурдагы экскурсияга келишкен балдар көргөн

Metadata: exact_match=0.765625, prefix=96, length=128, termination=max_new_tokens, repetition=0.289062, invalid_decode=False

### 11. Data index `10`

**Prompt:** п чыгышты, биз балдар бул музейге дүйнөлүк тарых контекстинде өзүнүн тарыхын тереңирээк билүү �

**Reference:** �чүн дагы бир нече жолу келет деп ишенебиз. (Сүрөттөрдү сүрөт баянынан көрүүгө болот) Темир жол вокзалынын имаратындагы шып Биздин окуучуларыбыз темир жол вокзалынын имаратына киришип, шыптасындагы 1934-жылы вен

**Generated:** �чүн дагы бир нече жолу келет деп ишенебиз. (Сүрөттөрдү сүрөт баянынан көрүүгө болот) Темир жол вокзалынын имаратындагы шып Биздин окуучуларыбыз темир жол вокзалынын имаратына киришип, шыптасындагы 1934-жылы вен

Metadata: exact_match=1.000000, prefix=128, length=128, termination=max_new_tokens, repetition=0.445312, invalid_decode=True

### 12. Data index `11`

**Prompt:** ожомкулга, Курманжан Даткага, искусство ишмерлерине жана жакында эле кыргыз-казак достугун даңазалаган Абайга коюлган эстеликтерди жана б

**Reference:** ашка көптөгөн кызыктуу жайларды кыдырып, “Нөлдүк километрдин” желесин изилдедик. Ала-Тоо аянтындагы Кыргызстандын желегин кайтарган Улуттук гвардиянын күзөт кароолунун алмашуу аземи балдарга өзгөчө таасир эт

**Generated:** ашка көптөгөн кызыктуу жайларды кыдырып, “Нөлдүк километрдин” желесин изилдедик. Ала-Тоо аянтындагы Кыргызстандын желегин кайтарган Улуттук гвардиянын күзөт кароолунун алмашуу аземи балдарга өзгөчө таасир эт

Metadata: exact_match=1.000000, prefix=128, length=128, termination=max_new_tokens, repetition=0.398438, invalid_decode=False

### 13. Data index `12`

**Prompt:**  тартуулады, алар биздин өлкөнүн Эгемендик күнүнө карата даярдап жаткан бийин бийлеп беришти. Мугалим Фари

**Reference:** да Кенжеалиевна Суранчиева ырдайт Көрүүчүлөр аябай курсант болду. Биздин балдарыбыз да карап турбай БХУнун сахнасына чыгып, өздөрү даярдап келген номерлерин көрсөттү. Ат-Башыдан келген кичинекей кыздар бийлеп беришсе, ы

**Generated:** да Кенжеалиевна Суранчиева ырдайт Көрүүчүлөр аябай курсант болду. Биздин балдарыбыз да карап турбай БХУнун сахнасына чыгып, өздөрү даярдап келген номерлерин көрсөттү. Ат-Башыдан келген кичинекей кыздар бийлеп беришсе, ы

Metadata: exact_match=1.000000, prefix=128, length=128, termination=max_new_tokens, repetition=0.382812, invalid_decode=False

### 14. Data index `13`

**Prompt:** ен саякаттоосу жөнүндө: › билим берүү › биринчи мугалим сыйлыгы › балдардын саякаты Мугалимдердин саякаттары жө

**Reference:** нүндө: билим берүү › биринчи мугалим сыйлыгы › мугалимдин саякаты Сынактардын жүрүшү жөнүндө: билим берүү › биринчи мугалим сынагы › сынактын күндөлүгү Сынактардан алынган сүрөт баяндары: › билим

**Generated:** нүндө: билим берүү › биринчи мугалим сыйлыгы › мугалимдин саякаты Сынактардын жүрүшү жөнүндө: билим берүү › биринчи мугалим сынагы › сынактын күндөлүгү Сынактардан алынган сүрөт баяндары: › билим

Metadata: exact_match=1.000000, prefix=128, length=128, termination=max_new_tokens, repetition=0.656250, invalid_decode=False

### 15. Data index `14`

**Prompt:**  ГАНА КАРАП КАЛГАН ДЕЙСИЗБИ? АНДАЙ Э... 2022. БАЛДАРДЫН САЯКАТЫ БАЛДАР ЖАНА ЧОҢДОР ҮЧҮ

**Reference:** Н ЧОҢ КУБАНЫЧ!... 2022. БАЛДАРДЫН САЯКАТЫ – БАЛДАР ҮЧҮН ЖАНА ЧОҢДОР ҮЧҮН КУБАН... 2018. КҮН НУРЛУУ БАЛДАР КҮНӨСТҮҮ БИШКЕКТЕ: БАТКЕНДЕГИ ЖАҢЫ-Б... МУГАЛИМДИН САЯ

**Generated:** Н ЧОҢ КУБАНЫЧ!... 2022. БАЛДАРДЫН САЯКАТЫ – БАЛДАР ҮЧҮН ЖАНА ЧОҢДОР ҮЧҮН КУБАН... 2018. КҮН НУРЛУУ БАЛДАР КҮНӨСТҮҮ БИШКЕКТЕ: БАТКЕНДЕГИ ЖАҢЫ-Б... МУГАЛИМДИН САЯ

Metadata: exact_match=1.000000, prefix=128, length=128, termination=max_new_tokens, repetition=0.617188, invalid_decode=False

### 16. Data index `15`

**Prompt:** ТИН МУГАЛИМИ, ФОН... 18.10.2016 Андан ары окуңуз… БАЛДАРДЫН САЯКАТЫ 2023. АТ-БАШЫ РАЙОНУНУН А

**Reference:** ЧА-КАЙЫҢДЫ АЙЫЛЫНДАГЫ ОЙ-ТЕРСКЕН МЕ... 09.11.2023 2023. КАРА-КУЛЖА ЖАНА КАРА-ТАЛААДАГЫ БАШТАЛГЫЧ КЛАССТАРДЫН О... 09.09.2023 2022. «БИРИНЧИ МУГАЛИМ» 10-СЫНАГЫНЫН ЛАУРЕАТТАРЫНЫ

**Generated:** ЧА-КАЙЫҢДЫ АЙЫЛЫНДАГЫ ОЙ-ТЕРСКЕН МЕ... 09.11.2023 2023. КАРА-КУЛЖА ЖАНА КАРА-ТАЛААДАГЫ БАШТАЛГЫЧ КЛАССТАРДЫН САЯКАТЫ 09.09.2023 2022. «БИРИНЧИ МУГАЛИМ» 10-СЫНАГЫНЫН ЛАУРЕАТТ

Metadata: exact_match=0.632812, prefix=80, length=128, termination=max_new_tokens, repetition=0.554688, invalid_decode=False

## Decision

Cache modes disagree reproducibly, so the primary diagnosis is an invalid or incomplete generation cache path. The discrepancy is deterministic within each mode, not random ROCm sampling noise: 10 of 12 records matched, while frozen-base Kyrgyz prompt 02 diverged at generated token 6 and LoRA English prompt 02 diverged at token 22.

The cache result means previous `use_cache=True` free-generation results cannot be treated as fully reliable model behavior until this path is fixed. The isolated micro-overfit was run with `use_cache=False` and removes document mixing, cross-sequence labels, and full-sequence prompt loss. It reached continuation CE `0.014994` / PPL `1.015107`; 12 of 16 training continuations matched all 128 tokens, with mean token match `0.889160` and mean prefix match `113.5` tokens. This demonstrates that the LoRA objective, masked labels, data loading, optimizer, and model integration can memorize the isolated training examples; they are not categorically unable to learn.

The remaining four examples are partial matches rather than a perfect 16/16 memorization result, and one contains an invalid decode, so this is strong but not flawless overfit evidence. Once cache parity is repaired, a no-cache versus cache-heldout comparison is still required before attributing the earlier heldout degeneration to generalization or corpus quality.
