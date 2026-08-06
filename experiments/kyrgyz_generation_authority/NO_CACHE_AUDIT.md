# Authoritative No-Cache Generation Audit

Every comparison below uses `use_cache=False`, greedy decoding, and 64 new tokens.

## Metrics

```json
{
  "model": "/workspace/model/real_SmolLM3-3B",
  "device": "cuda",
  "decoder": {
    "use_cache": false,
    "do_sample": false,
    "max_new_tokens": 64
  },
  "prompts_per_source": 5,
  "conditions": [
    "frozen_base",
    "axis_step_0200",
    "axis_step_2000",
    "lora_step_0200"
  ],
  "records": 60,
  "aggregate": {
    "frozen_base/heldout_kyrgyz": {
      "samples": 5,
      "average_generated_length": 64.0,
      "eos_rate": 0.0,
      "mean_repetition_rate": 0.8,
      "invalid_decode_count": 2
    },
    "frozen_base/kyrgyz_flores": {
      "samples": 5,
      "average_generated_length": 64.0,
      "eos_rate": 0.0,
      "mean_repetition_rate": 0.740625,
      "invalid_decode_count": 1
    },
    "frozen_base/english_ood": {
      "samples": 5,
      "average_generated_length": 64.0,
      "eos_rate": 0.0,
      "mean_repetition_rate": 0.465625,
      "invalid_decode_count": 0
    },
    "axis_step_0200/heldout_kyrgyz": {
      "samples": 5,
      "average_generated_length": 64.0,
      "eos_rate": 0.0,
      "mean_repetition_rate": 0.634375,
      "invalid_decode_count": 1
    },
    "axis_step_0200/kyrgyz_flores": {
      "samples": 5,
      "average_generated_length": 64.0,
      "eos_rate": 0.0,
      "mean_repetition_rate": 0.703125,
      "invalid_decode_count": 0
    },
    "axis_step_0200/english_ood": {
      "samples": 5,
      "average_generated_length": 64.0,
      "eos_rate": 0.0,
      "mean_repetition_rate": 0.4375,
      "invalid_decode_count": 0
    },
    "axis_step_2000/heldout_kyrgyz": {
      "samples": 5,
      "average_generated_length": 64.0,
      "eos_rate": 0.0,
      "mean_repetition_rate": 0.3625,
      "invalid_decode_count": 0
    },
    "axis_step_2000/kyrgyz_flores": {
      "samples": 5,
      "average_generated_length": 64.0,
      "eos_rate": 0.0,
      "mean_repetition_rate": 0.528125,
      "invalid_decode_count": 2
    },
    "axis_step_2000/english_ood": {
      "samples": 5,
      "average_generated_length": 64.0,
      "eos_rate": 0.0,
      "mean_repetition_rate": 0.98125,
      "invalid_decode_count": 0
    },
    "lora_step_0200/heldout_kyrgyz": {
      "samples": 5,
      "average_generated_length": 64.0,
      "eos_rate": 0.0,
      "mean_repetition_rate": 0.771875,
      "invalid_decode_count": 1
    },
    "lora_step_0200/kyrgyz_flores": {
      "samples": 5,
      "average_generated_length": 64.0,
      "eos_rate": 0.0,
      "mean_repetition_rate": 0.634375,
      "invalid_decode_count": 2
    },
    "lora_step_0200/english_ood": {
      "samples": 5,
      "average_generated_length": 64.0,
      "eos_rate": 0.0,
      "mean_repetition_rate": 0.434375,
      "invalid_decode_count": 0
    }
  },
  "wall_time_s": 92.50399276101962,
  "axis_checkpoint_hashes": {
    "host": "root@36.150.116.206:31101",
    "checkpoint_dir": "/workspace/GLT/experiments/kyrgyz_generation/checkpoints",
    "checkpoints": {
      "axis_step_0050.pt": "dd1bc07aa5e51ae51de70004312180fa2f55c6da5472185303f071a5b1e9d3a3",
      "axis_step_0100.pt": "8ea8302a18f18f1ac0158f6c945d137ef8ff2a55ae937ad0ef58ec971c29e05b",
      "axis_step_0200.pt": "dd27003b2578b7c40ddd50b7f11fda1f77721717e3afba25c3da21921b17c80d",
      "axis_step_0500.pt": "6cba5747b4859d5016e6cb51fb5c7941308cb360d92130cfb2e2871275778664",
      "axis_step_1000.pt": "fd6f22b1291affae0be26d7505d8b71718c5d125f015fd18bf16ea19cd3c8621",
      "axis_step_2000.pt": "f57bc713ea6e6f9b4c72c70d5ed47326c2ba3ac4bbee690c474211390b833a8e"
    }
  },
  "lora_artifact_hashes": {
    "artifact_dir": "/workspace/GLT/experiments/kyrgyz_lora_control/lora_step_0200",
    "files": {
      "README.md": "4e72ac623b80ff3f7a10bdff4fc2a47f65127c899e120f733835b1ec5e413b4d",
      "adapter_config.json": "9d55ccd95159b3e50d3291880baeb1b9ed39dd264c26908b2d917f1f6c585476",
      "adapter_model.safetensors": "6ac2ca1421fe5c8f7b15724933d62df955f162cd2491c82b22e6b8ef7d1f006d"
    },
    "sha256_manifest": "7701b62ca178c72e77e92c95584dfef0edf2e799562968daf3c3d9d74bd19766"
  }
}
```

## Complete Outputs

### 1. `frozen_base` / `heldout_kyrgyz_00`

**Prompt:** Министрлер кабинетинин төрагасынын орун басары Эдил Байсалов Фейсбук барагында «Бириккен демократиялык кыймылынын» билдир

**Reference:** үүсүнө жооп берди. Кыймыл 13-апрелде Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген болчу. «Оппозиция деп бул досторубузду атоо туура эмес. Равшан Бабыр уулу 2,5 миң добуш алган президенттик шайлоодо. Клара айымдыкы дурусурак: 1% — 14 миң добуш. Анан кайдагы оппозиция боло алышмак эле? Активист дегилечи сураныч. Ошол туура болот!» — деп жазды Байсалов. Байсалов кыймылдын мүчөлөрүн оппозиция деп эсептебесин билдирди. «Уят эле бирок. Саясатчы менен артист сезиши керек сахнадан качан түшүштү. Эптеп эле чырлап, кыпчылып, өрдөк жокто чулдук бий болуу аракети. Оппозиция болот, азыр деле бар. Бирок бул активисттер эмес. Бузуку деп айтпайын. Бирок Кумтөрдүн 100% элибизге кайтарылышына БУУ баш болуп, дүйнөлүк коомчулук суктанып, алкыш айтып атышса, куттуктап, бу

**Generated:** е келгесен келтиркен, «Бириккен демократиялык кыймылынын» билдире келгесен келтиркен, «Бириккен демократиялык кыймы

Metadata: length=64, termination=max_new_tokens, repetition=0.562500, invalid_decode=False

### 2. `frozen_base` / `heldout_kyrgyz_01`

**Prompt:** лар тескери каршы чыгып аткандары бузукулук эмей эмне», — дейт чиновник. Байсаловдун билдирүүсүнө «Бириккен дем

**Reference:** ократиялык кыймылдын» мүчөсү, саясатчы Клара Сооронкулова дагы жооп кайтарды. «Байсаловду сөгө албаганыма эч качан өкүнгөн эмесмин. Кайда, кайсы кеменин артынан “жибереримди” билбей турам. Байкуш. Премьер-министрликке добуш берүү менен шайланса бир жөн эле», — деп жазды Сооронкулова Фейсбук барагында. Оппозиционерлердин билдирүүсү «Бириккен демократиялык кыймыл» 13-апрелде «Кумтөр: коррупциялык жана экологиялык тобокелчиликтер» деген темада өткөн жыйында Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген. Саясатчылар Кумтөрдү улутташтыруунун зарылдыгы тууралуу кайрылуу даярдашканын билдиришкен. Алардын пикиринде, канадалык компания менен келишим түзүү — Кыргызстан үчүн утулуш болот, анткени өлкө «Центеррадагы» акцияларынан баш тартып, экологиялык

**Generated:** ократиялар, демократиялар, демократиялар, демократиялар, демократиялар, демократиялар, демократиялар, демократиялар, демократиялар, демок

Metadata: length=64, termination=max_new_tokens, repetition=0.890625, invalid_decode=False

### 3. `frozen_base` / `heldout_kyrgyz_02`

**Prompt:**  чыгымдарды өз мойнуна алып жатат. «Бириккен демократиялык кыймыл» — бул кыргыз оппозициясынын жаңы түзү

**Reference:** мү, кыймылдын түзүлгөнү 31-мартта белгилүү болгон. Анын курамында Равшан Жээнбеков, Клара Сооронкулова, Бектур Асанов, Мамбетжунус Абылов, Кеңешбек Дүйшөбаев өңдүү саясатчылар, ошондой эле укук коргоочулар жана активисттер бар. Кыймылдын координатору — Мамбетжунус Абылов.<|end_of_text|>Макензи, Маккензи – Канаданын түндүк-батышындагы дарыя. Чоң Эрксиз көлүнөн башталып, Макензи ойдуңу аркылуу агат да Түндүк Муз океандын Бофорт деңизине куят. Өзүнүн Узундугу 1770 кмдей, Пис-Ривер менен (Финли дарыясынын башынан) 4250 км, алабынын аянты (Чоң Эрксиз көлүнүн алабына кирген Эрксиз, Пис-Ривер жана Атабаска дарыясынын системасы менен) 1804 миң км2. Ири куймалары: Лиард, Арктик-Ред-Ривер, Пил (сол), Чоң Аюулуу (оң). Жээги өтө саздак, карагай токою менен

**Generated:** лүүнүнүнүнүнүнүнүнүнүнүнүнүнүнүнүнүнүнүнүн�

Metadata: length=64, termination=max_new_tokens, repetition=0.937500, invalid_decode=True

### 4. `frozen_base` / `heldout_kyrgyz_03`

**Prompt:**  капталган. Кар, жамгыр сууларынан курулат. Жаз жана жай мезгилдеринде ташкындайт. Орточо чыгымы 14 миң м3/сек. Октябр

**Reference:** дан июнга чейин тоңот. Кеме жолунун узундугу 2200 км. Боюнда Форт-Провиденс, Форт-Норман, Норман-Уэлс, Аклавик, Инувик калктуу пункттары жайгашкан. Дарыя, европалыктардын ичинен алгачкы жолу (1789) ал аркылуу сүзүп, Түндүк Муз океанга чыккан шотланддык көпөс, саякатчы – А. Макензинин урматынан аталган. Колдонулган адабияттар[түзөтүү | булагын түзөтүү] “Кыргызстан” улуттук энциклопедиясы: 5-том. Башкы редактору Асанов Ү. А. К 97. Б.: Мамлекеттик тил жана энциклопедия борбору, 2014. илл. ISBN 978 9967-14-111-7 "https://ky.wikipedia.org/w/index.php?title=Макензи&oldid=319180" булагынан алынды Категория: Канада Дарыялар Жашыруун категория: Pages using ISBN magic links Бул барак соңку жолу 5 май 2018, саат 12:39 өзгөртүлгөн. Текст Creative Commons Attribution-ShareAlike лицензиясынын алкагында жеткиликтүү; кошумча шарттар колдонулушу мүмкүн. Кеңири маалымат ү

**Generated:** ү 2012 жана 1-н саңтүнү 1-н саңтүнүн 1-н саңтүнүн 1-н саңтүн�

Metadata: length=64, termination=max_new_tokens, repetition=0.781250, invalid_decode=True

### 5. `frozen_base` / `heldout_kyrgyz_04`

**Prompt:** чүн Колдонуу шарттарын караңыз.<|end_of_text|>“Ысык-Көл” эл аралык аэропорту бүгүн, 20-ноябрда, 100 000-жүрг

**Reference:** үнчүсүн тейледи. Юбилейлик жүргүнчү 2021-жылдын 20-ноябрында “Аэрофлот” авиакомпаниясынын каттамы менен Москва шаарына учуп жаткан Жумабаева Чолпон болду. Бул тууралуу «Манас» эл аралык аэропортунун басма сөз кызматы билдирди. Бул — тарыхый окуя , себеби «Ысык-Көл» аэропорту алгачкы жолу 100 миң кишилик чектен ашты. Тактап айтсак, аэропорт 2021 — жылы 1000ден ашык рейсти тейлесе, бүгүн 100 000-жүргүнчүсүн тейледи. «Аймактардагы аэропортторубуз да өнүгүп, 100 миң кишилик чекке жеткенине кубанып турабыз Бул биз үчүн жакшы көрсөткүч. Биздин жүргүнчүбүздү чын дилимден куттуктайм” деп белгиледи «Манас» эл аралык аэропортунун башкармалыгынын төрагасы Бакыт Шаршеев. Өз кезегинде, «Ысык-Көл» эл аралык аэропортунун башчысы Алмаз Абдрахманов жүргү

**Generated:** үүн жерүүнээс өлөөнөөөөөөөөөөөөөөөөөөөө

Metadata: length=64, termination=max_new_tokens, repetition=0.828125, invalid_decode=False

### 6. `frozen_base` / `kyrgyz_flores_00`

**Prompt:** Ал мурун "диабет оорусу менен ооруган, ал эми азыр оорудан айыккан 4 айлык чычкан бар деп кошумчалады".<|end_of_text|>Канаданын Диабет Ас

**Reference:** социациясынын илимий жана клиникалык бөлүмүнүн төрагасы жана Жаңы Шотландиянын Галифекс шаарындагы Далхаузи университетинин медицина илимдеринин профессору, доктор Эхуд Ур бул изилдөөнүн башталышы гана экенин эскертти.<|end_of_text|>Ал диабетти айыктыруу мүмкүнчүлүгүнөн айрым эксперттердей эле күмөн санайт жана бул натыйжалардын 1-типтеги диабет менен ооруган адамдарга тиешеси жок экенин белгилейт.<|end_of_text|>Дүйшөмбү күнү Швед академиясынын адабият боюнча Нобель комитетинин туруктуу катчысы Сара Даниус Швециядагы Sveriges Radio программасында 2016-жылдын адабияты боюнча Нобель сыйлыгын алуу жөнүндө Боб Дилан менен түздөн-түз байланыша албаган комитет аны менен байланышуу аракетин токтотконун жалпыга ачык билдирди.<|end_of_text|>"Учурда эч нерсе кылбай эле турабыз. Мен анын эң жакын кызматташына чалып, электрондук каттарды жөнөттү

**Generated:** оси (КДА) 2016-2017 гадында 1,4 млн. километр жердиндеги кызыл айыккан 1,2 млн. километр жердиндеги кызыл ай

Metadata: length=64, termination=max_new_tokens, repetition=0.421875, invalid_decode=False

### 7. `frozen_base` / `kyrgyz_flores_01`

**Prompt:** п, абдан жагымдуу жоопторду алдым. Бул албетте, азырынча жетиштүү", — деди Даниус.<|end_of_text|>Мындан мурда Ring компаниясынын башкы

**Reference:**  директору Жейми Симиноф компания анын гаражындагы дүкөндөн эшиктин шыңгыраганы угулбаганда иштей баштаганын белгилеген.<|end_of_text|>Ал WiFi менен иштеген эшик коңгуроосун жасаганын айтты.<|end_of_text|>Симиноффтун айтымында, ал 2013-жылы Shark Tank шоусунда чечим чыгарган топтор стартапты каржылоодон баш тарткан учурда чагылдырылгандан кийин, сатуулардын саны жогорулаган.<|end_of_text|>Симинофф 2017-жылдын аягында QVC соода телеканалына чыкты.<|end_of_text|>Ринг атаандаш коопсуздук компаниясы болгон ADT корпорациясы менен болгон соттук процессти жөнгө салган.<|end_of_text|>Бир эксперименталдык вакцина Эбола вирусунан болгон өлүмдү азайта алчудай, бирок бүгүнкү күнгө чейин инфекцияны дарылоого бир дагы дары ылайык көрсөтүлө элек.<|end_of_text|>ZMapp, антидене коктейли, бул жаатта өзүн көптөн үмүткөр кылган, бирок расмий изилдөөлөр көрсөткөндөй анын өлүм

**Generated:** нда, өзгөрөнөнөнөнөнөнөнөнөнөнөнөнөнөнөнөнөнө

Metadata: length=64, termination=max_new_tokens, repetition=0.859375, invalid_decode=False

### 8. `frozen_base` / `kyrgyz_flores_02`

**Prompt:** дү алдын алуудагы пайдасы аз.<|end_of_text|>PALM сыноосунда, ZMapp көзөмөлдөө катары кызмат кылган, атап айтканда о

**Reference:** кумуштуулар аны баштапкы план катары колдонушуп, ага үч башка дарылоону салыштырышкан.<|end_of_text|>USA Gymnastics АКШнын Олимпиадалык комитетинин катын колдойт жана бардык спортчуларыбыз үчүн коопсуз чөйрөнү тартуулоого олимпиада үй-бүлөсүнүн бардык муктаждыктарынын милдеттенмесин өзүнө алат.<|end_of_text|>Американын олимпиада комитети (USOC) тастыктамадан баш тарткандын ордуна, уюмубуздун ичинде олуттуу өзгөрүүлөрдү жасоо аркылуу клуб менен спортсмендерибиздин кызыкчылыктарын тейлей алабыз деген билдирүүсүнө макулбуз.<|end_of_text|>Америка Гимнастикасы Ларри Насардан аман калгандар көп убакыт бою байкалбаганын жана пропорцияны кыянаттык менен пайдаланууну көрсөтүшү мүмкүн болгон көз карандысыз иликтөөнү колдойт жана керектүү өзгөртүүлөрдү киргизиши мүмкүн.<|end_of_text|>АКШ Гимнастикасы жана Олимпиадалы

**Generated:** ны өзгөрөнөнөнөнөнөнөнөнөнөнөнөнөнөнөнөнөнөн�

Metadata: length=64, termination=max_new_tokens, repetition=0.875000, invalid_decode=True

### 9. `frozen_base` / `kyrgyz_flores_03`

**Prompt:** к Комитети бир эле максатты көздөшөт — гимнастиканы жана спорттун башка түрлөрүн спортчулар үчүн мүм

**Reference:** күн болушунча коопсуз кылып, кыялдарынын коопсуз, позитивдүү жана укуктуу чөйрөдө ишке ашуусун камсыз кылуу.<|end_of_text|>1960-жж. Бржезинский Джон Кеннединин кеңешчиси болуп, кийинчерээк Линдон Джонсондун кол алдында иштеген.<|end_of_text|>1976-ж. шайлоодо Картердин тышкы саясатты алып баруу боюнча кеңешчиси, кийинчерээк 1977-1981-жж. Генри Киссинджердин ордуна Улуттук коопсуздук боюнча кеңешчи (NSA) болгон.<|end_of_text|>Улуттук башкаруу агенттиги катары ал Картерге төмөнкүдөй жалпы көйгөйлөрдү дипломаттык түрдө чечүүгө жардам берген: 1978-жылдагы Кэмп-Дэвид макулдашуусу, 1970-жылдын аягында америка-кытай мамилелерин калыбына келтирүү, 1979-жылы Иранды кризиске алып келген Иран төңкөрүшү, 1979-жылы Советтер союзунун Афганистанга кол салышы.<|end_of_text|>Эмма Стоун жана Раян Гослинг катышкан фильм негизги категориялардын ба

**Generated:** көнөнөнөнөнөнөнөнөнөнөнөнөнөнөнөнөнөнөнөнөн

Metadata: length=64, termination=max_new_tokens, repetition=0.937500, invalid_decode=False

### 10. `frozen_base` / `kyrgyz_flores_04`

**Prompt:** арында номинация алган.<|end_of_text|>Гослинг жана Стоун эң мыкты актер жана эң мыкты актриса номинацияларын алышты.<|end_of_text|>Башка номинацияларга эң

**Reference:**  мыкты сүрөт, режиссер, оператордук иш, костюмдар дизайны, тасмаларды монтаждоо, түпнуска саундтрек, өндүрүш дизайны, үн монтаждоо, үндөрдү аралаштыруу жана өзгөчө сценарий кирет.<|end_of_text|>Кинодогу "Угуу" (Кыялданган акылы жоктор) жана "Жылдыздардын шаары" деген эки ыр мыкты оригиналдуу ыр номинациясын алышты. Башка студиялардан өзгөчөлөнүп, Lionsgate студиясы 26 номинацияга ээ болду.<|end_of_text|>АКШнын президенти Дональд Трамп кечки жекшембиде пресс-катчы аркылуу жасаган билдирүүсүндө, АКШ аскерлери Сириядан чыгарын билдирди.<|end_of_text|>Бул жаңылык Трамп Түркиянын президенти Режеп Тайип Эрдоган менен телефондо жолугушкандан кийин билдирилди.<|end_of_text|>Билдирүүдө айтылгандай Европа мамлекеттери мекенине кайтарууну четке кагып, колго түшкөн ИГИЛ согушкерлерин кайтарууну Түркия колго алганын билдирди.<|end_of_text|>Бул айрым дин

**Generated:**  мыкты актер жана эң мыкты актриса номинацияларын алышты. Эң мыкты актер жана эң мыкты актриса номинацияларын алышты. Эң

Metadata: length=64, termination=max_new_tokens, repetition=0.609375, invalid_decode=False

### 11. `frozen_base` / `english_ood_00`

**Prompt:** Barovier&Toso is a Venetian company with a vocation and culture that is international, creating solutions of decorative lighting in Murano glass, all characterised by a personal and unique style. ​ Lighting solutions destined not only to furnish ambiances, but also to create emotions, transmit positivity, making each room

**Reference:**  unique. ​ Traditions and the future Murano glass crafting tradition, of which we have always been absolute protagonists, is for us the starting point from which to propose ideas, solutions, innovative and original projects that have allowed us to hand down values through the century old history of the founders’ family.<|end_of_text|>Beckton locksmith services are offered by us, locks in the City. A true 24 hour emergency service, we have the solution anytime Day or Night. When you call our Beckton Locksmith service we speak with an actual locksmith. Daniel or myself Paul. With a combined 25 years experience and knowledge in all kinds of locks and hardware. We will be more then happy to attend your property for a free survey Beckton Locksmith services are used by domestic and commercial customers. We have customers within the retail parks that we carry out many typical locksmithing duties. We also have a retail shop for customers to come to us if they wish to. Where you can purchase anything from novelty keys through to Locks and associated hardware. If you need us to attend to fit the item we would be more then willing as we are the local Beckton Locksmith. Call Locks in the City today on 07921 000672 for a fast and friendly response. Our locksmiths are fully insured and are members of the Master Locksmiths Association (MLA). The team has been inspected by the MLA. We are fully qualified & fully insured. Consequently, you can have complete peace of mind when your Locks in the City representative arrives at your address. You know that they will be qualified to do the work. We use non destructive entry methods where ever possible in emergency lock out situations.<|end_of_text|>match preview: The Exiles head to rugby lane and sutton & epsom, for the first time - in round 6 - London Welsh Rugby Club Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh Home Teams Men’s Men’s Rugby Performance Squad Women & Girls Women

**Generated:**  a unique place. The company was founded in 1980 by the brothers Giuseppe and Vittorio Barovier, who have always been passionate about the art of glass. The company has been able to grow and develop thanks to the collaboration of the Toso brothers, who have been working in the field of lighting

Metadata: length=64, termination=max_new_tokens, repetition=0.281250, invalid_decode=False

### 12. `frozen_base` / `english_ood_01`

**Prompt:** ’s Rugby Women’s Squad Girls Youth & Minis Youth Rugby Mini Rugby Fixtures Latest News News & Media News Videos Gallery Podcast Membership Supporters club About Us Join Us Travel And Events The 400 Club Careers Hub Hub News The Hub The Idea Success Stories The Mentoring Experience Business Partners Contact The Hub Events & Hospitality LWRFC

**Reference:**  Choir Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh match preview: The Exiles head to rugby lane and sutton & epsom, for the first time – in round 6 News October 9, 2022 Following the disappointment of Round 5 and yet another top 4 side that Welsh will feel they should have beaten, Round 6 is yet another first time fixture for the London Welsh faithful. The Exiles fourth away trip of the season takes them to Ewell in Sutton and a side that Welsh meet for the first time in their history (something to which we have become accustom to since project reset). Sutton & Epsom has a similar longevity to that of London Welsh, playing their first matches in the 1883/84 season – the season prior to the founding of London Welsh. In recent seasons Sutton have found the going tough after promotion in 18/19 season from Level 5 to level 4 and National 2 South. Sutton are one of the sides to count themselves unlucky from the RFU calculations resulting from the 80% played season of covid riven 19/20. Sutton finished the played season 3rd from bottom and were relegated with no opportunity to complete the season and try to reel in 4th from bottom Westcliff. That said, Sutton were 14 points adrift of safety at that stage. In the first full season post Covid, Sutton finished 11th in South Premier last season (4th from bottom) and comfortably clear of Brighton in 12th. Although early in the season both sides will see this Round 6 fixture as crucial: Sutton currently sit just one place above Welsh in the league (9th) and with an identical tally of won 1 lost 4 (Sutton with 2 more bonus points to their name). Welsh will feel that they could and perhaps should have won all 4 of their previous fixtures, having taken 3 of the current top

**Generated:**  2024-25 Season 2024-25 Season 2024-25 Season 2024-25 Season 2024-25 Season 2024-25 Season 2024-25 Season 2024-25 Season 2024-25 Season 2024-25 Season 2024-

Metadata: length=64, termination=max_new_tokens, repetition=0.906250, invalid_decode=False

### 13. `frozen_base` / `english_ood_02`

**Prompt:** 4 sides all the way. That all said, Welsh need league points in order to get their season properly underway. It is well accepted that Welsh have been victims of an almost unprecedented injury crisis in the early rounds, an issue that is only now seeing signs of abating. In the latest injury and availability news: Top

**Reference:**  try scorer Sion Cowdy remains on the side-lines; he is on track to return in Round 7 and is with the Ireland 7s squad in Spain this weekend. There will be a fitness test this week for Loosehead Griff Whitson after he too missed Round 6 following injury in Round 5. Will Ponty at second row suffered a concussion in Round 5 and will be undergo HIA protocol assessment this week as he clears the mandatory 3 weeks on the side-lines. Former Premiership star Ben West is almost ready for his first game of the season but Round 6 will come just that bit too early. In positive news: Matt Pollen returns at centre for the first time since Round 2, Wales u18 man Rhys Howells also returns in the centre for potentially his 53rd cap and following his wedding last weekend. Hywel Williams also returns at to contest the starting spot at 9 having been unavailable for round 5. Tom Williams played 80 minutes for the VETS (as their u35 player) against Wasps on Friday last as he continues his 6 month return to fitness, he may be in contention for a first game of the season. James Smith (broken shoulder) and Jack Anderson (dislocated knee) remain the long term absentees. Welsh could jump as high as 7th win a win in Round 6 and will want to continue their excellent form and growth that we have seen since round 4. Director of Men’s Rugby Cai Griffiths said of the game ” We have another big week in training this week with the squad returning to some form of normality and continuity, given the injuries that we have had. We expect this Round 6 match to be another tough contest with Sutton & Epsom coming close in many of their games so far this season. We have certainly felt like we are not far off where we need to be in the past few weeks and with a more settled team the squad will be keen to prove some points’. No doubt eyes will be on some of the big performers from the last few rounds: Samoan Kiwi Tololima Saavainaea has be excellent in his first three games and grabbed his

**Generated:**  scorer and captain, Rhys Webb, is out for the season after suffering a cruciate ligament injury in the 2nd round. The Welsh manager, Paul Williams, has also been forced to make do with a reduced squad, with the likes of Rhys Griffiths, Rhys Griffiths, and Rhys

Metadata: length=64, termination=max_new_tokens, repetition=0.296875, invalid_decode=False

### 14. `frozen_base` / `english_ood_03`

**Prompt:**  first try in Round 5 against Havant. Ben Bodinham has been excellent having moved to 8 from the second row, following the loss of Jack Anderson at 8 for the rest of the season. Supporters and the squad will welcome the return of Matt Pollen at centre and hopes will be high that he

**Reference:**  makes a big impact on first game back since Round 2. Top points scorer Osian McAvoy has been excellent following a switch this season from 10 to fullback and his talent will not stand alone in a talented back line. Kick off will be 3pm at Sutton & Epsom RFC, Cuddington Court, Rugby Lane. Details will be released this week on the supporters transport to match. We hope to see the large travelling exile army with us as always. London Welsh’s next home game will come in Round 7 on 22 October when Bracknell RFC visit Old Deer Park. #COYW share: Post navigation Previous Post Match report: bp win despite the odds for LWWRFC Next Post Lasyl and Location:Live sponsor London Welsh Colts Academy As they Take on Monmouth school You May Also Like News November 11, 2022 the sad loss of Don mcmanus It is with great sadness that the London Welsh RFC community this week has learnt of the death in… News December 9, 2020 The Boys are back in town London Welsh emerge from Lockdown 2.0 with the go-ahead to resume competitive contact rugby! With some tasty fixtures already…<|end_of_text|>He'll be out for eight weeks which means he will still be available for the All Blacks World Cup campaign in France in September. Moody's the fourth prop the Crusaders have lost during their title defence. The Super Rugby Pacific side has already lost Fletcher Newell, George Bower and Finlay Brewis to injuries. The Crusaders hope to add another prop to their roster in the near future. 34 year old Moody missed much of last season with an ACL injury. He played at both the 2015 and 2019 World Cups. Meanwhile, Blues coach Leon Macdonald has confirmed first-five Beauden Barrett will miss Saturday's game against the Hurricanes. Barrett has a heel injury picked up in the last round against the Reds in Brisbane. "He's on his All Blacks rest this week which coincides with his stitches in his foot, it's hard to know for sure but we're pretty confident he's going to be okay [for next week]." Macdonald says Barrett needed to

**Generated:**  can help the team to a better finish than last season. The squad is looking forward to the new season and the challenge ahead.

The squad for the new season is as follows:

1. Ben Bodinham
2. Jack Anderson
3. Matt Pollen
4. Tom Hargreaves
5.

Metadata: length=64, termination=max_new_tokens, repetition=0.265625, invalid_decode=False

### 15. `frozen_base` / `english_ood_04`

**Prompt:**  take his mandatory All Blacks rest this round or next but his injury forced their hand. The fourth placed Blues meet the fifth placed Hurricanes at Eden Park in a battle for fourth spot or higher on the Super Rugby Pacific table with a home playoff game on the line.<|end_of_text|>A galaxy of stars will headline the NRL grand final entertainment

**Reference:**  while Brisbane will pay tribute to rugby league royalty in the lead-up to the game with King George Square renamed King Wally Square. Grand Final Week officially kicked off on Sunday evening with Brisbane City Hall, King George Tower and the Hilton lit up with iconic images from the season, tributes to Penrith and South Sydney, the game’s fans and a thank you to the Queensland community for their hospitality. King George Square in the Brisbane CBD is being renamed King Wally Square in honour of Queensland Immortal Wally Lewis as part of a two-day NRL Fan Fest on Thursday and Friday, culminating with the appearance of both teams on Friday afternoon. "I think the name choice is fantastic," Lewis said with a laugh on Sunday night. "A lot of people have been asking me if I think it's a little bit of a giggle fest. Yeah, I do but after King George passed away I think it was 83 years ago, it was put in his honour. I don't have any plans to be passing away soon. "To have an honour like this certainly is wonderful but I'm fully aware it's only going to be going on whilst the rugby league grand final is being put in the planners and it's going to be going back to King George Square immediately after the game is finished." Lewis said he was extremely proud of the people of Queensland for offering their support to the NRL players in recent months during difficult circumstances." Next Sunday's pre-game entertainment at Suncorp Stadium will have a Queensland theme with home-grown star Kate Miller-Heidke to sing the national anthem and the award-winning singer-songwriter will also lead the entertainment with Ian Moss, respected across the globe as one of Australia’s most iconic musicians. Globe-trotting DJs and producers The Stafford Brothers, multi-Platinum DJ, producer and live instrumentalist Timmy Trumpet, internationally acclaimed didgeridoo player and composer William Barton and a 40-piece Orchestra will complete the grand final entertainment which will also include a spectacular light show. NRL chief executive Andrew Abdo said the week-long celebration was a fitting way to thank Queensland for supporting the Telstra Premiership in 2021 after 13 teams relocated to the Sunshine State in July

**Generated:**  at the Sydney Cricket Ground on Saturday night. The NRL grand final will be the first to be held at the SCG since 2003 and will be the first to be held on a Saturday night since 2001. The grand final will be the first to be held at the SCG since 2003 and

Metadata: length=64, termination=max_new_tokens, repetition=0.578125, invalid_decode=False

### 16. `axis_step_0200` / `heldout_kyrgyz_00`

**Prompt:** Министрлер кабинетинин төрагасынын орун басары Эдил Байсалов Фейсбук барагында «Бириккен демократиялык кыймылынын» билдир

**Reference:** үүсүнө жооп берди. Кыймыл 13-апрелде Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген болчу. «Оппозиция деп бул досторубузду атоо туура эмес. Равшан Бабыр уулу 2,5 миң добуш алган президенттик шайлоодо. Клара айымдыкы дурусурак: 1% — 14 миң добуш. Анан кайдагы оппозиция боло алышмак эле? Активист дегилечи сураныч. Ошол туура болот!» — деп жазды Байсалов. Байсалов кыймылдын мүчөлөрүн оппозиция деп эсептебесин билдирди. «Уят эле бирок. Саясатчы менен артист сезиши керек сахнадан качан түшүштү. Эптеп эле чырлап, кыпчылып, өрдөк жокто чулдук бий болуу аракети. Оппозиция болот, азыр деле бар. Бирок бул активисттер эмес. Бузуку деп айтпайын. Бирок Кумтөрдүн 100% элибизге кайтарылышына БУУ баш болуп, дүйнөлүк коомчулук суктанып, алкыш айтып атышса, куттуктап, бу

**Generated:** ди. Байсалов Фейсбук барагында «Бириккен демократиялык кыймылынын» билдирди. Байсалов Фейсбук барагында «Бириккен демократ

Metadata: length=64, termination=max_new_tokens, repetition=0.453125, invalid_decode=False

### 17. `axis_step_0200` / `heldout_kyrgyz_01`

**Prompt:** лар тескери каршы чыгып аткандары бузукулук эмей эмне», — дейт чиновник. Байсаловдун билдирүүсүнө «Бириккен дем

**Reference:** ократиялык кыймылдын» мүчөсү, саясатчы Клара Сооронкулова дагы жооп кайтарды. «Байсаловду сөгө албаганыма эч качан өкүнгөн эмесмин. Кайда, кайсы кеменин артынан “жибереримди” билбей турам. Байкуш. Премьер-министрликке добуш берүү менен шайланса бир жөн эле», — деп жазды Сооронкулова Фейсбук барагында. Оппозиционерлердин билдирүүсү «Бириккен демократиялык кыймыл» 13-апрелде «Кумтөр: коррупциялык жана экологиялык тобокелчиликтер» деген темада өткөн жыйында Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген. Саясатчылар Кумтөрдү улутташтыруунун зарылдыгы тууралуу кайрылуу даярдашканын билдиришкен. Алардын пикиринде, канадалык компания менен келишим түзүү — Кыргызстан үчүн утулуш болот, анткени өлкө «Центеррадагы» акцияларынан баш тартып, экологиялык

**Generated:** илгенин» байланышында кыргызстандын түзөтүүчүлүгүнүн түзүүчүлүгүнү

Metadata: length=64, termination=max_new_tokens, repetition=0.625000, invalid_decode=False

### 18. `axis_step_0200` / `heldout_kyrgyz_02`

**Prompt:**  чыгымдарды өз мойнуна алып жатат. «Бириккен демократиялык кыймыл» — бул кыргыз оппозициясынын жаңы түзү

**Reference:** мү, кыймылдын түзүлгөнү 31-мартта белгилүү болгон. Анын курамында Равшан Жээнбеков, Клара Сооронкулова, Бектур Асанов, Мамбетжунус Абылов, Кеңешбек Дүйшөбаев өңдүү саясатчылар, ошондой эле укук коргоочулар жана активисттер бар. Кыймылдын координатору — Мамбетжунус Абылов.<|end_of_text|>Макензи, Маккензи – Канаданын түндүк-батышындагы дарыя. Чоң Эрксиз көлүнөн башталып, Макензи ойдуңу аркылуу агат да Түндүк Муз океандын Бофорт деңизине куят. Өзүнүн Узундугу 1770 кмдей, Пис-Ривер менен (Финли дарыясынын башынан) 4250 км, алабынын аянты (Чоң Эрксиз көлүнүн алабына кирген Эрксиз, Пис-Ривер жана Атабаска дарыясынын системасы менен) 1804 миң км2. Ири куймалары: Лиард, Арктик-Ред-Ривер, Пил (сол), Чоң Аюулуу (оң). Жээги өтө саздак, карагай токою менен

**Generated:** мдүн байланышы. Бул тууралуу өзүнчү кыргызстандын демилгенин байланышы. Бул тууралуу өзүнчү кы

Metadata: length=64, termination=max_new_tokens, repetition=0.531250, invalid_decode=False

### 19. `axis_step_0200` / `heldout_kyrgyz_03`

**Prompt:**  капталган. Кар, жамгыр сууларынан курулат. Жаз жана жай мезгилдеринде ташкындайт. Орточо чыгымы 14 миң м3/сек. Октябр

**Reference:** дан июнга чейин тоңот. Кеме жолунун узундугу 2200 км. Боюнда Форт-Провиденс, Форт-Норман, Норман-Уэлс, Аклавик, Инувик калктуу пункттары жайгашкан. Дарыя, европалыктардын ичинен алгачкы жолу (1789) ал аркылуу сүзүп, Түндүк Муз океанга чыккан шотланддык көпөс, саякатчы – А. Макензинин урматынан аталган. Колдонулган адабияттар[түзөтүү | булагын түзөтүү] “Кыргызстан” улуттук энциклопедиясы: 5-том. Башкы редактору Асанов Ү. А. К 97. Б.: Мамлекеттик тил жана энциклопедия борбору, 2014. илл. ISBN 978 9967-14-111-7 "https://ky.wikipedia.org/w/index.php?title=Макензи&oldid=319180" булагынан алынды Категория: Канада Дарыялар Жашыруун категория: Pages using ISBN magic links Бул барак соңку жолу 5 май 2018, саат 12:39 өзгөртүлгөн. Текст Creative Commons Attribution-ShareAlike лицензиясынын алкагында жеткиликтүү; кошумча шарттар колдонулушу мүмкүн. Кеңири маалымат ү

**Generated:** да өзүнүн түзүү жана түзүү жана түзүү жана түзүү жана түзүү жана тү

Metadata: length=64, termination=max_new_tokens, repetition=0.828125, invalid_decode=False

### 20. `axis_step_0200` / `heldout_kyrgyz_04`

**Prompt:** чүн Колдонуу шарттарын караңыз.<|end_of_text|>“Ысык-Көл” эл аралык аэропорту бүгүн, 20-ноябрда, 100 000-жүрг

**Reference:** үнчүсүн тейледи. Юбилейлик жүргүнчү 2021-жылдын 20-ноябрында “Аэрофлот” авиакомпаниясынын каттамы менен Москва шаарына учуп жаткан Жумабаева Чолпон болду. Бул тууралуу «Манас» эл аралык аэропортунун басма сөз кызматы билдирди. Бул — тарыхый окуя , себеби «Ысык-Көл» аэропорту алгачкы жолу 100 миң кишилик чектен ашты. Тактап айтсак, аэропорт 2021 — жылы 1000ден ашык рейсти тейлесе, бүгүн 100 000-жүргүнчүсүн тейледи. «Аймактардагы аэропортторубуз да өнүгүп, 100 миң кишилик чекке жеткенине кубанып турабыз Бул биз үчүн жакшы көрсөткүч. Биздин жүргүнчүбүздү чын дилимден куттуктайм” деп белгиледи «Манас» эл аралык аэропортунун башкармалыгынын төрагасы Бакыт Шаршеев. Өз кезегинде, «Ысык-Көл» эл аралык аэропортунун башчысы Алмаз Абдрахманов жүргү

**Generated:** үнчү түзөтүү жана 200 000-жүргүнчү түзөтүү түзөтүү түзөт�

Metadata: length=64, termination=max_new_tokens, repetition=0.734375, invalid_decode=True

### 21. `axis_step_0200` / `kyrgyz_flores_00`

**Prompt:** Ал мурун "диабет оорусу менен ооруган, ал эми азыр оорудан айыккан 4 айлык чычкан бар деп кошумчалады".<|end_of_text|>Канаданын Диабет Ас

**Reference:** социациясынын илимий жана клиникалык бөлүмүнүн төрагасы жана Жаңы Шотландиянын Галифекс шаарындагы Далхаузи университетинин медицина илимдеринин профессору, доктор Эхуд Ур бул изилдөөнүн башталышы гана экенин эскертти.<|end_of_text|>Ал диабетти айыктыруу мүмкүнчүлүгүнөн айрым эксперттердей эле күмөн санайт жана бул натыйжалардын 1-типтеги диабет менен ооруган адамдарга тиешеси жок экенин белгилейт.<|end_of_text|>Дүйшөмбү күнү Швед академиясынын адабият боюнча Нобель комитетинин туруктуу катчысы Сара Даниус Швециядагы Sveriges Radio программасында 2016-жылдын адабияты боюнча Нобель сыйлыгын алуу жөнүндө Боб Дилан менен түздөн-түз байланыша албаган комитет аны менен байланышуу аракетин токтотконун жалпыга ачык билдирди.<|end_of_text|>"Учурда эч нерсе кылбай эле турабыз. Мен анын эң жакын кызматташына чалып, электрондук каттарды жөнөттү

**Generated:** оциациясынын байланышы 2021-жылы өзүнчү байланышы өттү. Байланышы өттүнүн байланышы ө

Metadata: length=64, termination=max_new_tokens, repetition=0.609375, invalid_decode=False

### 22. `axis_step_0200` / `kyrgyz_flores_01`

**Prompt:** п, абдан жагымдуу жоопторду алдым. Бул албетте, азырынча жетиштүү", — деди Даниус.<|end_of_text|>Мындан мурда Ring компаниясынын башкы

**Reference:**  директору Жейми Симиноф компания анын гаражындагы дүкөндөн эшиктин шыңгыраганы угулбаганда иштей баштаганын белгилеген.<|end_of_text|>Ал WiFi менен иштеген эшик коңгуроосун жасаганын айтты.<|end_of_text|>Симиноффтун айтымында, ал 2013-жылы Shark Tank шоусунда чечим чыгарган топтор стартапты каржылоодон баш тарткан учурда чагылдырылгандан кийин, сатуулардын саны жогорулаган.<|end_of_text|>Симинофф 2017-жылдын аягында QVC соода телеканалына чыкты.<|end_of_text|>Ринг атаандаш коопсуздук компаниясы болгон ADT корпорациясы менен болгон соттук процессти жөнгө салган.<|end_of_text|>Бир эксперименталдык вакцина Эбола вирусунан болгон өлүмдү азайта алчудай, бирок бүгүнкү күнгө чейин инфекцияны дарылоого бир дагы дары ылайык көрсөтүлө элек.<|end_of_text|>ZMapp, антидене коктейли, бул жаатта өзүн көптөн үмүткөр кылган, бирок расмий изилдөөлөр көрсөткөндөй анын өлүм

**Generated:**  менен байланышын түзөтүү. Бул компаниянын байланышын түзөтүү, байланышын түзөтүү, бай

Metadata: length=64, termination=max_new_tokens, repetition=0.656250, invalid_decode=False

### 23. `axis_step_0200` / `kyrgyz_flores_02`

**Prompt:** дү алдын алуудагы пайдасы аз.<|end_of_text|>PALM сыноосунда, ZMapp көзөмөлдөө катары кызмат кылган, атап айтканда о

**Reference:** кумуштуулар аны баштапкы план катары колдонушуп, ага үч башка дарылоону салыштырышкан.<|end_of_text|>USA Gymnastics АКШнын Олимпиадалык комитетинин катын колдойт жана бардык спортчуларыбыз үчүн коопсуз чөйрөнү тартуулоого олимпиада үй-бүлөсүнүн бардык муктаждыктарынын милдеттенмесин өзүнө алат.<|end_of_text|>Американын олимпиада комитети (USOC) тастыктамадан баш тарткандын ордуна, уюмубуздун ичинде олуттуу өзгөрүүлөрдү жасоо аркылуу клуб менен спортсмендерибиздин кызыкчылыктарын тейлей алабыз деген билдирүүсүнө макулбуз.<|end_of_text|>Америка Гимнастикасы Ларри Насардан аман калгандар көп убакыт бою байкалбаганын жана пропорцияны кыянаттык менен пайдаланууну көрсөтүшү мүмкүн болгон көз карандысыз иликтөөнү колдойт жана керектүү өзгөртүүлөрдү киргизиши мүмкүн.<|end_of_text|>АКШ Гимнастикасы жана Олимпиадалы

**Generated:** шондой эле кыргызстандын жана биздин жана биздин кыргызстандын жана биздин кыргызстандын жана биздин кыргызстандын жана биздин кы

Metadata: length=64, termination=max_new_tokens, repetition=0.734375, invalid_decode=False

### 24. `axis_step_0200` / `kyrgyz_flores_03`

**Prompt:** к Комитети бир эле максатты көздөшөт — гимнастиканы жана спорттун башка түрлөрүн спортчулар үчүн мүм

**Reference:** күн болушунча коопсуз кылып, кыялдарынын коопсуз, позитивдүү жана укуктуу чөйрөдө ишке ашуусун камсыз кылуу.<|end_of_text|>1960-жж. Бржезинский Джон Кеннединин кеңешчиси болуп, кийинчерээк Линдон Джонсондун кол алдында иштеген.<|end_of_text|>1976-ж. шайлоодо Картердин тышкы саясатты алып баруу боюнча кеңешчиси, кийинчерээк 1977-1981-жж. Генри Киссинджердин ордуна Улуттук коопсуздук боюнча кеңешчи (NSA) болгон.<|end_of_text|>Улуттук башкаруу агенттиги катары ал Картерге төмөнкүдөй жалпы көйгөйлөрдү дипломаттык түрдө чечүүгө жардам берген: 1978-жылдагы Кэмп-Дэвид макулдашуусу, 1970-жылдын аягында америка-кытай мамилелерин калыбына келтирүү, 1979-жылы Иранды кризиске алып келген Иран төңкөрүшү, 1979-жылы Советтер союзунун Афганистанга кол салышы.<|end_of_text|>Эмма Стоун жана Раян Гослинг катышкан фильм негизги категориялардын ба

**Generated:** күнчүлүк. Бирок, бирок, бирок, бирок, бирок, бирок, бирок, бирок, бирок, бирок, бирок, бирок, бирок,

Metadata: length=64, termination=max_new_tokens, repetition=0.812500, invalid_decode=False

### 25. `axis_step_0200` / `kyrgyz_flores_04`

**Prompt:** арында номинация алган.<|end_of_text|>Гослинг жана Стоун эң мыкты актер жана эң мыкты актриса номинацияларын алышты.<|end_of_text|>Башка номинацияларга эң

**Reference:**  мыкты сүрөт, режиссер, оператордук иш, костюмдар дизайны, тасмаларды монтаждоо, түпнуска саундтрек, өндүрүш дизайны, үн монтаждоо, үндөрдү аралаштыруу жана өзгөчө сценарий кирет.<|end_of_text|>Кинодогу "Угуу" (Кыялданган акылы жоктор) жана "Жылдыздардын шаары" деген эки ыр мыкты оригиналдуу ыр номинациясын алышты. Башка студиялардан өзгөчөлөнүп, Lionsgate студиясы 26 номинацияга ээ болду.<|end_of_text|>АКШнын президенти Дональд Трамп кечки жекшембиде пресс-катчы аркылуу жасаган билдирүүсүндө, АКШ аскерлери Сириядан чыгарын билдирди.<|end_of_text|>Бул жаңылык Трамп Түркиянын президенти Режеп Тайип Эрдоган менен телефондо жолугушкандан кийин билдирилди.<|end_of_text|>Билдирүүдө айтылгандай Европа мамлекеттери мекенине кайтарууну четке кагып, колго түшкөн ИГИЛ согушкерлерин кайтарууну Түркия колго алганын билдирди.<|end_of_text|>Бул айрым дин

**Generated:**  мыкты актерын айырмалган. Бул эң мыкты актерын айырмалган. Бул эң мыкты актерын айырмалган. Бул эң мыкты ак

Metadata: length=64, termination=max_new_tokens, repetition=0.703125, invalid_decode=False

### 26. `axis_step_0200` / `english_ood_00`

**Prompt:** Barovier&Toso is a Venetian company with a vocation and culture that is international, creating solutions of decorative lighting in Murano glass, all characterised by a personal and unique style. ​ Lighting solutions destined not only to furnish ambiances, but also to create emotions, transmit positivity, making each room

**Reference:**  unique. ​ Traditions and the future Murano glass crafting tradition, of which we have always been absolute protagonists, is for us the starting point from which to propose ideas, solutions, innovative and original projects that have allowed us to hand down values through the century old history of the founders’ family.<|end_of_text|>Beckton locksmith services are offered by us, locks in the City. A true 24 hour emergency service, we have the solution anytime Day or Night. When you call our Beckton Locksmith service we speak with an actual locksmith. Daniel or myself Paul. With a combined 25 years experience and knowledge in all kinds of locks and hardware. We will be more then happy to attend your property for a free survey Beckton Locksmith services are used by domestic and commercial customers. We have customers within the retail parks that we carry out many typical locksmithing duties. We also have a retail shop for customers to come to us if they wish to. Where you can purchase anything from novelty keys through to Locks and associated hardware. If you need us to attend to fit the item we would be more then willing as we are the local Beckton Locksmith. Call Locks in the City today on 07921 000672 for a fast and friendly response. Our locksmiths are fully insured and are members of the Master Locksmiths Association (MLA). The team has been inspected by the MLA. We are fully qualified & fully insured. Consequently, you can have complete peace of mind when your Locks in the City representative arrives at your address. You know that they will be qualified to do the work. We use non destructive entry methods where ever possible in emergency lock out situations.<|end_of_text|>match preview: The Exiles head to rugby lane and sutton & epsom, for the first time - in round 6 - London Welsh Rugby Club Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh Home Teams Men’s Men’s Rugby Performance Squad Women & Girls Women

**Generated:**  a unique space. The company was founded in 1999 by the two brothers, Roberto and Massimo Barovier, who have always been passionate about glass and light. The name of the company is a tribute to the famous glassmakers of the 16th century, Angelo Barovier and Lorenzo Torosio

Metadata: length=64, termination=max_new_tokens, repetition=0.265625, invalid_decode=False

### 27. `axis_step_0200` / `english_ood_01`

**Prompt:** ’s Rugby Women’s Squad Girls Youth & Minis Youth Rugby Mini Rugby Fixtures Latest News News & Media News Videos Gallery Podcast Membership Supporters club About Us Join Us Travel And Events The 400 Club Careers Hub Hub News The Hub The Idea Success Stories The Mentoring Experience Business Partners Contact The Hub Events & Hospitality LWRFC

**Reference:**  Choir Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh match preview: The Exiles head to rugby lane and sutton & epsom, for the first time – in round 6 News October 9, 2022 Following the disappointment of Round 5 and yet another top 4 side that Welsh will feel they should have beaten, Round 6 is yet another first time fixture for the London Welsh faithful. The Exiles fourth away trip of the season takes them to Ewell in Sutton and a side that Welsh meet for the first time in their history (something to which we have become accustom to since project reset). Sutton & Epsom has a similar longevity to that of London Welsh, playing their first matches in the 1883/84 season – the season prior to the founding of London Welsh. In recent seasons Sutton have found the going tough after promotion in 18/19 season from Level 5 to level 4 and National 2 South. Sutton are one of the sides to count themselves unlucky from the RFU calculations resulting from the 80% played season of covid riven 19/20. Sutton finished the played season 3rd from bottom and were relegated with no opportunity to complete the season and try to reel in 4th from bottom Westcliff. That said, Sutton were 14 points adrift of safety at that stage. In the first full season post Covid, Sutton finished 11th in South Premier last season (4th from bottom) and comfortably clear of Brighton in 12th. Although early in the season both sides will see this Round 6 fixture as crucial: Sutton currently sit just one place above Welsh in the league (9th) and with an identical tally of won 1 lost 4 (Sutton with 2 more bonus points to their name). Welsh will feel that they could and perhaps should have won all 4 of their previous fixtures, having taken 3 of the current top

**Generated:**  LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC LWRFC L

Metadata: length=64, termination=max_new_tokens, repetition=0.953125, invalid_decode=False

### 28. `axis_step_0200` / `english_ood_02`

**Prompt:** 4 sides all the way. That all said, Welsh need league points in order to get their season properly underway. It is well accepted that Welsh have been victims of an almost unprecedented injury crisis in the early rounds, an issue that is only now seeing signs of abating. In the latest injury and availability news: Top

**Reference:**  try scorer Sion Cowdy remains on the side-lines; he is on track to return in Round 7 and is with the Ireland 7s squad in Spain this weekend. There will be a fitness test this week for Loosehead Griff Whitson after he too missed Round 6 following injury in Round 5. Will Ponty at second row suffered a concussion in Round 5 and will be undergo HIA protocol assessment this week as he clears the mandatory 3 weeks on the side-lines. Former Premiership star Ben West is almost ready for his first game of the season but Round 6 will come just that bit too early. In positive news: Matt Pollen returns at centre for the first time since Round 2, Wales u18 man Rhys Howells also returns in the centre for potentially his 53rd cap and following his wedding last weekend. Hywel Williams also returns at to contest the starting spot at 9 having been unavailable for round 5. Tom Williams played 80 minutes for the VETS (as their u35 player) against Wasps on Friday last as he continues his 6 month return to fitness, he may be in contention for a first game of the season. James Smith (broken shoulder) and Jack Anderson (dislocated knee) remain the long term absentees. Welsh could jump as high as 7th win a win in Round 6 and will want to continue their excellent form and growth that we have seen since round 4. Director of Men’s Rugby Cai Griffiths said of the game ” We have another big week in training this week with the squad returning to some form of normality and continuity, given the injuries that we have had. We expect this Round 6 match to be another tough contest with Sutton & Epsom coming close in many of their games so far this season. We have certainly felt like we are not far off where we need to be in the past few weeks and with a more settled team the squad will be keen to prove some points’. No doubt eyes will be on some of the big performers from the last few rounds: Samoan Kiwi Tololima Saavainaea has be excellent in his first three games and grabbed his

**Generated:**  scorer and captain, James McArthur, is out for the season after suffering a serious knee injury. The 26-year-old was a key player for the Welsh in the first two rounds, scoring 10 goals in 12 appearances. McArthur was a key player for the Welsh in the first two rounds, scoring

Metadata: length=64, termination=max_new_tokens, repetition=0.421875, invalid_decode=False

### 29. `axis_step_0200` / `english_ood_03`

**Prompt:**  first try in Round 5 against Havant. Ben Bodinham has been excellent having moved to 8 from the second row, following the loss of Jack Anderson at 8 for the rest of the season. Supporters and the squad will welcome the return of Matt Pollen at centre and hopes will be high that he

**Reference:**  makes a big impact on first game back since Round 2. Top points scorer Osian McAvoy has been excellent following a switch this season from 10 to fullback and his talent will not stand alone in a talented back line. Kick off will be 3pm at Sutton & Epsom RFC, Cuddington Court, Rugby Lane. Details will be released this week on the supporters transport to match. We hope to see the large travelling exile army with us as always. London Welsh’s next home game will come in Round 7 on 22 October when Bracknell RFC visit Old Deer Park. #COYW share: Post navigation Previous Post Match report: bp win despite the odds for LWWRFC Next Post Lasyl and Location:Live sponsor London Welsh Colts Academy As they Take on Monmouth school You May Also Like News November 11, 2022 the sad loss of Don mcmanus It is with great sadness that the London Welsh RFC community this week has learnt of the death in… News December 9, 2020 The Boys are back in town London Welsh emerge from Lockdown 2.0 with the go-ahead to resume competitive contact rugby! With some tasty fixtures already…<|end_of_text|>He'll be out for eight weeks which means he will still be available for the All Blacks World Cup campaign in France in September. Moody's the fourth prop the Crusaders have lost during their title defence. The Super Rugby Pacific side has already lost Fletcher Newell, George Bower and Finlay Brewis to injuries. The Crusaders hope to add another prop to their roster in the near future. 34 year old Moody missed much of last season with an ACL injury. He played at both the 2015 and 2019 World Cups. Meanwhile, Blues coach Leon Macdonald has confirmed first-five Beauden Barrett will miss Saturday's game against the Hurricanes. Barrett has a heel injury picked up in the last round against the Reds in Brisbane. "He's on his All Blacks rest this week which coincides with his stitches in his foot, it's hard to know for sure but we're pretty confident he's going to be okay [for next week]." Macdonald says Barrett needed to

**Generated:**  can help the team to a better finish than last season. The squad is looking forward to the new season and the challenge ahead.

The squad for the new season is as follows:

1. Jack Anderson
2. Ben Bodinham
3. Matt Pollen
4. Tom Hargreaves
5.

Metadata: length=64, termination=max_new_tokens, repetition=0.265625, invalid_decode=False

### 30. `axis_step_0200` / `english_ood_04`

**Prompt:**  take his mandatory All Blacks rest this round or next but his injury forced their hand. The fourth placed Blues meet the fifth placed Hurricanes at Eden Park in a battle for fourth spot or higher on the Super Rugby Pacific table with a home playoff game on the line.<|end_of_text|>A galaxy of stars will headline the NRL grand final entertainment

**Reference:**  while Brisbane will pay tribute to rugby league royalty in the lead-up to the game with King George Square renamed King Wally Square. Grand Final Week officially kicked off on Sunday evening with Brisbane City Hall, King George Tower and the Hilton lit up with iconic images from the season, tributes to Penrith and South Sydney, the game’s fans and a thank you to the Queensland community for their hospitality. King George Square in the Brisbane CBD is being renamed King Wally Square in honour of Queensland Immortal Wally Lewis as part of a two-day NRL Fan Fest on Thursday and Friday, culminating with the appearance of both teams on Friday afternoon. "I think the name choice is fantastic," Lewis said with a laugh on Sunday night. "A lot of people have been asking me if I think it's a little bit of a giggle fest. Yeah, I do but after King George passed away I think it was 83 years ago, it was put in his honour. I don't have any plans to be passing away soon. "To have an honour like this certainly is wonderful but I'm fully aware it's only going to be going on whilst the rugby league grand final is being put in the planners and it's going to be going back to King George Square immediately after the game is finished." Lewis said he was extremely proud of the people of Queensland for offering their support to the NRL players in recent months during difficult circumstances." Next Sunday's pre-game entertainment at Suncorp Stadium will have a Queensland theme with home-grown star Kate Miller-Heidke to sing the national anthem and the award-winning singer-songwriter will also lead the entertainment with Ian Moss, respected across the globe as one of Australia’s most iconic musicians. Globe-trotting DJs and producers The Stafford Brothers, multi-Platinum DJ, producer and live instrumentalist Timmy Trumpet, internationally acclaimed didgeridoo player and composer William Barton and a 40-piece Orchestra will complete the grand final entertainment which will also include a spectacular light show. NRL chief executive Andrew Abdo said the week-long celebration was a fitting way to thank Queensland for supporting the Telstra Premiership in 2021 after 13 teams relocated to the Sunshine State in July

**Generated:**  at the Sydney Cricket Ground on Saturday night. The NRL grand final entertainment will include a live performance by the legendary rock band AC/DC, a tribute to the late great Johnny O’Neil, and a special appearance by the legendary Australian singer Olivia Newton-John. The NRL grand final entertainment will be held at the Sydney

Metadata: length=64, termination=max_new_tokens, repetition=0.281250, invalid_decode=False

### 31. `axis_step_2000` / `heldout_kyrgyz_00`

**Prompt:** Министрлер кабинетинин төрагасынын орун басары Эдил Байсалов Фейсбук барагында «Бириккен демократиялык кыймылынын» билдир

**Reference:** үүсүнө жооп берди. Кыймыл 13-апрелде Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген болчу. «Оппозиция деп бул досторубузду атоо туура эмес. Равшан Бабыр уулу 2,5 миң добуш алган президенттик шайлоодо. Клара айымдыкы дурусурак: 1% — 14 миң добуш. Анан кайдагы оппозиция боло алышмак эле? Активист дегилечи сураныч. Ошол туура болот!» — деп жазды Байсалов. Байсалов кыймылдын мүчөлөрүн оппозиция деп эсептебесин билдирди. «Уят эле бирок. Саясатчы менен артист сезиши керек сахнадан качан түшүштү. Эптеп эле чырлап, кыпчылып, өрдөк жокто чулдук бий болуу аракети. Оппозиция болот, азыр деле бар. Бирок бул активисттер эмес. Бузуку деп айтпайын. Бирок Кумтөрдүн 100% элибизге кайтарылышына БУУ баш болуп, дүйнөлүк коомчулук суктанып, алкыш айтып атышса, куттуктап, бу

**Generated:** үүлөрүнөн кийин кайрылды. Билдирүүнүн кеңири маалыматында, ал өзүнүн кызматы

Metadata: length=64, termination=max_new_tokens, repetition=0.484375, invalid_decode=False

### 32. `axis_step_2000` / `heldout_kyrgyz_01`

**Prompt:** лар тескери каршы чыгып аткандары бузукулук эмей эмне», — дейт чиновник. Байсаловдун билдирүүсүнө «Бириккен дем

**Reference:** ократиялык кыймылдын» мүчөсү, саясатчы Клара Сооронкулова дагы жооп кайтарды. «Байсаловду сөгө албаганыма эч качан өкүнгөн эмесмин. Кайда, кайсы кеменин артынан “жибереримди” билбей турам. Байкуш. Премьер-министрликке добуш берүү менен шайланса бир жөн эле», — деп жазды Сооронкулова Фейсбук барагында. Оппозиционерлердин билдирүүсү «Бириккен демократиялык кыймыл» 13-апрелде «Кумтөр: коррупциялык жана экологиялык тобокелчиликтер» деген темада өткөн жыйында Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген. Саясатчылар Кумтөрдү улутташтыруунун зарылдыгы тууралуу кайрылуу даярдашканын билдиришкен. Алардын пикиринде, канадалык компания менен келишим түзүү — Кыргызстан үчүн утулуш болот, анткени өлкө «Центеррадагы» акцияларынан баш тартып, экологиялык

**Generated:** ократиясы» коомдук уюмунун тышкы саясаттын негизи болуп келген. Бул күнү Бишкек шаарында «Бириккен демократиясы» ко

Metadata: length=64, termination=max_new_tokens, repetition=0.265625, invalid_decode=False

### 33. `axis_step_2000` / `heldout_kyrgyz_02`

**Prompt:**  чыгымдарды өз мойнуна алып жатат. «Бириккен демократиялык кыймыл» — бул кыргыз оппозициясынын жаңы түзү

**Reference:** мү, кыймылдын түзүлгөнү 31-мартта белгилүү болгон. Анын курамында Равшан Жээнбеков, Клара Сооронкулова, Бектур Асанов, Мамбетжунус Абылов, Кеңешбек Дүйшөбаев өңдүү саясатчылар, ошондой эле укук коргоочулар жана активисттер бар. Кыймылдын координатору — Мамбетжунус Абылов.<|end_of_text|>Макензи, Маккензи – Канаданын түндүк-батышындагы дарыя. Чоң Эрксиз көлүнөн башталып, Макензи ойдуңу аркылуу агат да Түндүк Муз океандын Бофорт деңизине куят. Өзүнүн Узундугу 1770 кмдей, Пис-Ривер менен (Финли дарыясынын башынан) 4250 км, алабынын аянты (Чоң Эрксиз көлүнүн алабына кирген Эрксиз, Пис-Ривер жана Атабаска дарыясынын системасы менен) 1804 миң км2. Ири куймалары: Лиард, Арктик-Ред-Ривер, Пил (сол), Чоң Аюулуу (оң). Жээги өтө саздак, карагай токою менен

**Generated:** лгөн кыймылдын бири. Бул кыймылдын жетекчиси Акылбек Жапаров 2010-жылдан бери кыргызстандыктын кызыкчылыгын

Metadata: length=64, termination=max_new_tokens, repetition=0.343750, invalid_decode=False

### 34. `axis_step_2000` / `heldout_kyrgyz_03`

**Prompt:**  капталган. Кар, жамгыр сууларынан курулат. Жаз жана жай мезгилдеринде ташкындайт. Орточо чыгымы 14 миң м3/сек. Октябр

**Reference:** дан июнга чейин тоңот. Кеме жолунун узундугу 2200 км. Боюнда Форт-Провиденс, Форт-Норман, Норман-Уэлс, Аклавик, Инувик калктуу пункттары жайгашкан. Дарыя, европалыктардын ичинен алгачкы жолу (1789) ал аркылуу сүзүп, Түндүк Муз океанга чыккан шотланддык көпөс, саякатчы – А. Макензинин урматынан аталган. Колдонулган адабияттар[түзөтүү | булагын түзөтүү] “Кыргызстан” улуттук энциклопедиясы: 5-том. Башкы редактору Асанов Ү. А. К 97. Б.: Мамлекеттик тил жана энциклопедия борбору, 2014. илл. ISBN 978 9967-14-111-7 "https://ky.wikipedia.org/w/index.php?title=Макензи&oldid=319180" булагынан алынды Категория: Канада Дарыялар Жашыруун категория: Pages using ISBN magic links Бул барак соңку жолу 5 май 2018, саат 12:39 өзгөртүлгөн. Текст Creative Commons Attribution-ShareAlike лицензиясынын алкагында жеткиликтүү; кошумча шарттар колдонулушу мүмкүн. Кеңири маалымат ү

**Generated:** да өткөн жаңылыктарда өзгөрүлгөн жаңы чыгымдардын орточо чыгымы 16 миң м3/сек

Metadata: length=64, termination=max_new_tokens, repetition=0.468750, invalid_decode=False

### 35. `axis_step_2000` / `heldout_kyrgyz_04`

**Prompt:** чүн Колдонуу шарттарын караңыз.<|end_of_text|>“Ысык-Көл” эл аралык аэропорту бүгүн, 20-ноябрда, 100 000-жүрг

**Reference:** үнчүсүн тейледи. Юбилейлик жүргүнчү 2021-жылдын 20-ноябрында “Аэрофлот” авиакомпаниясынын каттамы менен Москва шаарына учуп жаткан Жумабаева Чолпон болду. Бул тууралуу «Манас» эл аралык аэропортунун басма сөз кызматы билдирди. Бул — тарыхый окуя , себеби «Ысык-Көл» аэропорту алгачкы жолу 100 миң кишилик чектен ашты. Тактап айтсак, аэропорт 2021 — жылы 1000ден ашык рейсти тейлесе, бүгүн 100 000-жүргүнчүсүн тейледи. «Аймактардагы аэропортторубуз да өнүгүп, 100 миң кишилик чекке жеткенине кубанып турабыз Бул биз үчүн жакшы көрсөткүч. Биздин жүргүнчүбүздү чын дилимден куттуктайм” деп белгиледи «Манас» эл аралык аэропортунун башкармалыгынын төрагасы Бакыт Шаршеев. Өз кезегинде, «Ысык-Көл» эл аралык аэропортунун башчысы Алмаз Абдрахманов жүргү

**Generated:** үн алып келген. Бул тууралуу эл аралык аэропорттук комитетинин төрагасы Акылбек Бекбеков билдирди. Аэропорттогу кызматк

Metadata: length=64, termination=max_new_tokens, repetition=0.250000, invalid_decode=False

### 36. `axis_step_2000` / `kyrgyz_flores_00`

**Prompt:** Ал мурун "диабет оорусу менен ооруган, ал эми азыр оорудан айыккан 4 айлык чычкан бар деп кошумчалады".<|end_of_text|>Канаданын Диабет Ас

**Reference:** социациясынын илимий жана клиникалык бөлүмүнүн төрагасы жана Жаңы Шотландиянын Галифекс шаарындагы Далхаузи университетинин медицина илимдеринин профессору, доктор Эхуд Ур бул изилдөөнүн башталышы гана экенин эскертти.<|end_of_text|>Ал диабетти айыктыруу мүмкүнчүлүгүнөн айрым эксперттердей эле күмөн санайт жана бул натыйжалардын 1-типтеги диабет менен ооруган адамдарга тиешеси жок экенин белгилейт.<|end_of_text|>Дүйшөмбү күнү Швед академиясынын адабият боюнча Нобель комитетинин туруктуу катчысы Сара Даниус Швециядагы Sveriges Radio программасында 2016-жылдын адабияты боюнча Нобель сыйлыгын алуу жөнүндө Боб Дилан менен түздөн-түз байланыша албаган комитет аны менен байланышуу аракетин токтотконун жалпыга ачык билдирди.<|end_of_text|>"Учурда эч нерсе кылбай эле турабыз. Мен анын эң жакын кызматташына чалып, электрондук каттарды жөнөттү

**Generated:** социациясынын башкы башчысы Дэвид Кэлли 1-октябрда Канаданын диабеттик бейтаптарына жана диабеттик бейтаптардын жетекчилери менен ж

Metadata: length=64, termination=max_new_tokens, repetition=0.312500, invalid_decode=False

### 37. `axis_step_2000` / `kyrgyz_flores_01`

**Prompt:** п, абдан жагымдуу жоопторду алдым. Бул албетте, азырынча жетиштүү", — деди Даниус.<|end_of_text|>Мындан мурда Ring компаниясынын башкы

**Reference:**  директору Жейми Симиноф компания анын гаражындагы дүкөндөн эшиктин шыңгыраганы угулбаганда иштей баштаганын белгилеген.<|end_of_text|>Ал WiFi менен иштеген эшик коңгуроосун жасаганын айтты.<|end_of_text|>Симиноффтун айтымында, ал 2013-жылы Shark Tank шоусунда чечим чыгарган топтор стартапты каржылоодон баш тарткан учурда чагылдырылгандан кийин, сатуулардын саны жогорулаган.<|end_of_text|>Симинофф 2017-жылдын аягында QVC соода телеканалына чыкты.<|end_of_text|>Ринг атаандаш коопсуздук компаниясы болгон ADT корпорациясы менен болгон соттук процессти жөнгө салган.<|end_of_text|>Бир эксперименталдык вакцина Эбола вирусунан болгон өлүмдү азайта алчудай, бирок бүгүнкү күнгө чейин инфекцияны дарылоого бир дагы дары ылайык көрсөтүлө элек.<|end_of_text|>ZMapp, антидене коктейли, бул жаатта өзүн көптөн үмүткөр кылган, бирок расмий изилдөөлөр көрсөткөндөй анын өлүм

**Generated:**  башчысы Бекзат Кадырбеков кыргызстандыктардын кызыкчылыгын коргоп жатканын айтты. Бул тууралуу компаниянын басма с

Metadata: length=64, termination=max_new_tokens, repetition=0.296875, invalid_decode=False

### 38. `axis_step_2000` / `kyrgyz_flores_02`

**Prompt:** дү алдын алуудагы пайдасы аз.<|end_of_text|>PALM сыноосунда, ZMapp көзөмөлдөө катары кызмат кылган, атап айтканда о

**Reference:** кумуштуулар аны баштапкы план катары колдонушуп, ага үч башка дарылоону салыштырышкан.<|end_of_text|>USA Gymnastics АКШнын Олимпиадалык комитетинин катын колдойт жана бардык спортчуларыбыз үчүн коопсуз чөйрөнү тартуулоого олимпиада үй-бүлөсүнүн бардык муктаждыктарынын милдеттенмесин өзүнө алат.<|end_of_text|>Американын олимпиада комитети (USOC) тастыктамадан баш тарткандын ордуна, уюмубуздун ичинде олуттуу өзгөрүүлөрдү жасоо аркылуу клуб менен спортсмендерибиздин кызыкчылыктарын тейлей алабыз деген билдирүүсүнө макулбуз.<|end_of_text|>Америка Гимнастикасы Ларри Насардан аман калгандар көп убакыт бою байкалбаганын жана пропорцияны кыянаттык менен пайдаланууну көрсөтүшү мүмкүн болгон көз карандысыз иликтөөнү колдойт жана керектүү өзгөртүүлөрдү киргизиши мүмкүн.<|end_of_text|>АКШ Гимнастикасы жана Олимпиадалы

**Generated:** орукана көрүүчүлөрүнүн көзөмөлүнүн алдын алуусуна жана алардын оорусуна көз�

Metadata: length=64, termination=max_new_tokens, repetition=0.625000, invalid_decode=True

### 39. `axis_step_2000` / `kyrgyz_flores_03`

**Prompt:** к Комитети бир эле максатты көздөшөт — гимнастиканы жана спорттун башка түрлөрүн спортчулар үчүн мүм

**Reference:** күн болушунча коопсуз кылып, кыялдарынын коопсуз, позитивдүү жана укуктуу чөйрөдө ишке ашуусун камсыз кылуу.<|end_of_text|>1960-жж. Бржезинский Джон Кеннединин кеңешчиси болуп, кийинчерээк Линдон Джонсондун кол алдында иштеген.<|end_of_text|>1976-ж. шайлоодо Картердин тышкы саясатты алып баруу боюнча кеңешчиси, кийинчерээк 1977-1981-жж. Генри Киссинджердин ордуна Улуттук коопсуздук боюнча кеңешчи (NSA) болгон.<|end_of_text|>Улуттук башкаруу агенттиги катары ал Картерге төмөнкүдөй жалпы көйгөйлөрдү дипломаттык түрдө чечүүгө жардам берген: 1978-жылдагы Кэмп-Дэвид макулдашуусу, 1970-жылдын аягында америка-кытай мамилелерин калыбына келтирүү, 1979-жылы Иранды кризиске алып келген Иран төңкөрүшү, 1979-жылы Советтер союзунун Афганистанга кол салышы.<|end_of_text|>Эмма Стоун жана Раян Гослинг катышкан фильм негизги категориялардын ба

**Generated:** күнчүлүк берүү. Бул боюнча бир эле күндөрүнүн көрсөтмөсүнөн кийин, �

Metadata: length=64, termination=max_new_tokens, repetition=0.562500, invalid_decode=True

### 40. `axis_step_2000` / `kyrgyz_flores_04`

**Prompt:** арында номинация алган.<|end_of_text|>Гослинг жана Стоун эң мыкты актер жана эң мыкты актриса номинацияларын алышты.<|end_of_text|>Башка номинацияларга эң

**Reference:**  мыкты сүрөт, режиссер, оператордук иш, костюмдар дизайны, тасмаларды монтаждоо, түпнуска саундтрек, өндүрүш дизайны, үн монтаждоо, үндөрдү аралаштыруу жана өзгөчө сценарий кирет.<|end_of_text|>Кинодогу "Угуу" (Кыялданган акылы жоктор) жана "Жылдыздардын шаары" деген эки ыр мыкты оригиналдуу ыр номинациясын алышты. Башка студиялардан өзгөчөлөнүп, Lionsgate студиясы 26 номинацияга ээ болду.<|end_of_text|>АКШнын президенти Дональд Трамп кечки жекшембиде пресс-катчы аркылуу жасаган билдирүүсүндө, АКШ аскерлери Сириядан чыгарын билдирди.<|end_of_text|>Бул жаңылык Трамп Түркиянын президенти Режеп Тайип Эрдоган менен телефондо жолугушкандан кийин билдирилди.<|end_of_text|>Билдирүүдө айтылгандай Европа мамлекеттери мекенине кайтарууну четке кагып, колго түшкөн ИГИЛ согушкерлерин кайтарууну Түркия колго алганын билдирди.<|end_of_text|>Бул айрым дин

**Generated:**  мыкты акыркы сааттын акыркы сааттын акыркы сааттын акыркы сааттын акыркы сааттын акыркы сааттын ак

Metadata: length=64, termination=max_new_tokens, repetition=0.843750, invalid_decode=False

### 41. `axis_step_2000` / `english_ood_00`

**Prompt:** Barovier&Toso is a Venetian company with a vocation and culture that is international, creating solutions of decorative lighting in Murano glass, all characterised by a personal and unique style. ​ Lighting solutions destined not only to furnish ambiances, but also to create emotions, transmit positivity, making each room

**Reference:**  unique. ​ Traditions and the future Murano glass crafting tradition, of which we have always been absolute protagonists, is for us the starting point from which to propose ideas, solutions, innovative and original projects that have allowed us to hand down values through the century old history of the founders’ family.<|end_of_text|>Beckton locksmith services are offered by us, locks in the City. A true 24 hour emergency service, we have the solution anytime Day or Night. When you call our Beckton Locksmith service we speak with an actual locksmith. Daniel or myself Paul. With a combined 25 years experience and knowledge in all kinds of locks and hardware. We will be more then happy to attend your property for a free survey Beckton Locksmith services are used by domestic and commercial customers. We have customers within the retail parks that we carry out many typical locksmithing duties. We also have a retail shop for customers to come to us if they wish to. Where you can purchase anything from novelty keys through to Locks and associated hardware. If you need us to attend to fit the item we would be more then willing as we are the local Beckton Locksmith. Call Locks in the City today on 07921 000672 for a fast and friendly response. Our locksmiths are fully insured and are members of the Master Locksmiths Association (MLA). The team has been inspected by the MLA. We are fully qualified & fully insured. Consequently, you can have complete peace of mind when your Locks in the City representative arrives at your address. You know that they will be qualified to do the work. We use non destructive entry methods where ever possible in emergency lock out situations.<|end_of_text|>match preview: The Exiles head to rugby lane and sutton & epsom, for the first time - in round 6 - London Welsh Rugby Club Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh Home Teams Men’s Men’s Rugby Performance Squad Women & Girls Women

**Generated:** ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

Metadata: length=64, termination=max_new_tokens, repetition=0.984375, invalid_decode=False

### 42. `axis_step_2000` / `english_ood_01`

**Prompt:** ’s Rugby Women’s Squad Girls Youth & Minis Youth Rugby Mini Rugby Fixtures Latest News News & Media News Videos Gallery Podcast Membership Supporters club About Us Join Us Travel And Events The 400 Club Careers Hub Hub News The Hub The Idea Success Stories The Mentoring Experience Business Partners Contact The Hub Events & Hospitality LWRFC

**Reference:**  Choir Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh match preview: The Exiles head to rugby lane and sutton & epsom, for the first time – in round 6 News October 9, 2022 Following the disappointment of Round 5 and yet another top 4 side that Welsh will feel they should have beaten, Round 6 is yet another first time fixture for the London Welsh faithful. The Exiles fourth away trip of the season takes them to Ewell in Sutton and a side that Welsh meet for the first time in their history (something to which we have become accustom to since project reset). Sutton & Epsom has a similar longevity to that of London Welsh, playing their first matches in the 1883/84 season – the season prior to the founding of London Welsh. In recent seasons Sutton have found the going tough after promotion in 18/19 season from Level 5 to level 4 and National 2 South. Sutton are one of the sides to count themselves unlucky from the RFU calculations resulting from the 80% played season of covid riven 19/20. Sutton finished the played season 3rd from bottom and were relegated with no opportunity to complete the season and try to reel in 4th from bottom Westcliff. That said, Sutton were 14 points adrift of safety at that stage. In the first full season post Covid, Sutton finished 11th in South Premier last season (4th from bottom) and comfortably clear of Brighton in 12th. Although early in the season both sides will see this Round 6 fixture as crucial: Sutton currently sit just one place above Welsh in the league (9th) and with an identical tally of won 1 lost 4 (Sutton with 2 more bonus points to their name). Welsh will feel that they could and perhaps should have won all 4 of their previous fixtures, having taken 3 of the current top

**Generated:** , but,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

Metadata: length=64, termination=max_new_tokens, repetition=0.968750, invalid_decode=False

### 43. `axis_step_2000` / `english_ood_02`

**Prompt:** 4 sides all the way. That all said, Welsh need league points in order to get their season properly underway. It is well accepted that Welsh have been victims of an almost unprecedented injury crisis in the early rounds, an issue that is only now seeing signs of abating. In the latest injury and availability news: Top

**Reference:**  try scorer Sion Cowdy remains on the side-lines; he is on track to return in Round 7 and is with the Ireland 7s squad in Spain this weekend. There will be a fitness test this week for Loosehead Griff Whitson after he too missed Round 6 following injury in Round 5. Will Ponty at second row suffered a concussion in Round 5 and will be undergo HIA protocol assessment this week as he clears the mandatory 3 weeks on the side-lines. Former Premiership star Ben West is almost ready for his first game of the season but Round 6 will come just that bit too early. In positive news: Matt Pollen returns at centre for the first time since Round 2, Wales u18 man Rhys Howells also returns in the centre for potentially his 53rd cap and following his wedding last weekend. Hywel Williams also returns at to contest the starting spot at 9 having been unavailable for round 5. Tom Williams played 80 minutes for the VETS (as their u35 player) against Wasps on Friday last as he continues his 6 month return to fitness, he may be in contention for a first game of the season. James Smith (broken shoulder) and Jack Anderson (dislocated knee) remain the long term absentees. Welsh could jump as high as 7th win a win in Round 6 and will want to continue their excellent form and growth that we have seen since round 4. Director of Men’s Rugby Cai Griffiths said of the game ” We have another big week in training this week with the squad returning to some form of normality and continuity, given the injuries that we have had. We expect this Round 6 match to be another tough contest with Sutton & Epsom coming close in many of their games so far this season. We have certainly felt like we are not far off where we need to be in the past few weeks and with a more settled team the squad will be keen to prove some points’. No doubt eyes will be on some of the big performers from the last few rounds: Samoan Kiwi Tololima Saavainaea has be excellent in his first three games and grabbed his

**Generated:** ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

Metadata: length=64, termination=max_new_tokens, repetition=0.984375, invalid_decode=False

### 44. `axis_step_2000` / `english_ood_03`

**Prompt:**  first try in Round 5 against Havant. Ben Bodinham has been excellent having moved to 8 from the second row, following the loss of Jack Anderson at 8 for the rest of the season. Supporters and the squad will welcome the return of Matt Pollen at centre and hopes will be high that he

**Reference:**  makes a big impact on first game back since Round 2. Top points scorer Osian McAvoy has been excellent following a switch this season from 10 to fullback and his talent will not stand alone in a talented back line. Kick off will be 3pm at Sutton & Epsom RFC, Cuddington Court, Rugby Lane. Details will be released this week on the supporters transport to match. We hope to see the large travelling exile army with us as always. London Welsh’s next home game will come in Round 7 on 22 October when Bracknell RFC visit Old Deer Park. #COYW share: Post navigation Previous Post Match report: bp win despite the odds for LWWRFC Next Post Lasyl and Location:Live sponsor London Welsh Colts Academy As they Take on Monmouth school You May Also Like News November 11, 2022 the sad loss of Don mcmanus It is with great sadness that the London Welsh RFC community this week has learnt of the death in… News December 9, 2020 The Boys are back in town London Welsh emerge from Lockdown 2.0 with the go-ahead to resume competitive contact rugby! With some tasty fixtures already…<|end_of_text|>He'll be out for eight weeks which means he will still be available for the All Blacks World Cup campaign in France in September. Moody's the fourth prop the Crusaders have lost during their title defence. The Super Rugby Pacific side has already lost Fletcher Newell, George Bower and Finlay Brewis to injuries. The Crusaders hope to add another prop to their roster in the near future. 34 year old Moody missed much of last season with an ACL injury. He played at both the 2015 and 2019 World Cups. Meanwhile, Blues coach Leon Macdonald has confirmed first-five Beauden Barrett will miss Saturday's game against the Hurricanes. Barrett has a heel injury picked up in the last round against the Reds in Brisbane. "He's on his All Blacks rest this week which coincides with his stitches in his foot, it's hard to know for sure but we're pretty confident he's going to be okay [for next week]." Macdonald says Barrett needed to

**Generated:** ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

Metadata: length=64, termination=max_new_tokens, repetition=0.984375, invalid_decode=False

### 45. `axis_step_2000` / `english_ood_04`

**Prompt:**  take his mandatory All Blacks rest this round or next but his injury forced their hand. The fourth placed Blues meet the fifth placed Hurricanes at Eden Park in a battle for fourth spot or higher on the Super Rugby Pacific table with a home playoff game on the line.<|end_of_text|>A galaxy of stars will headline the NRL grand final entertainment

**Reference:**  while Brisbane will pay tribute to rugby league royalty in the lead-up to the game with King George Square renamed King Wally Square. Grand Final Week officially kicked off on Sunday evening with Brisbane City Hall, King George Tower and the Hilton lit up with iconic images from the season, tributes to Penrith and South Sydney, the game’s fans and a thank you to the Queensland community for their hospitality. King George Square in the Brisbane CBD is being renamed King Wally Square in honour of Queensland Immortal Wally Lewis as part of a two-day NRL Fan Fest on Thursday and Friday, culminating with the appearance of both teams on Friday afternoon. "I think the name choice is fantastic," Lewis said with a laugh on Sunday night. "A lot of people have been asking me if I think it's a little bit of a giggle fest. Yeah, I do but after King George passed away I think it was 83 years ago, it was put in his honour. I don't have any plans to be passing away soon. "To have an honour like this certainly is wonderful but I'm fully aware it's only going to be going on whilst the rugby league grand final is being put in the planners and it's going to be going back to King George Square immediately after the game is finished." Lewis said he was extremely proud of the people of Queensland for offering their support to the NRL players in recent months during difficult circumstances." Next Sunday's pre-game entertainment at Suncorp Stadium will have a Queensland theme with home-grown star Kate Miller-Heidke to sing the national anthem and the award-winning singer-songwriter will also lead the entertainment with Ian Moss, respected across the globe as one of Australia’s most iconic musicians. Globe-trotting DJs and producers The Stafford Brothers, multi-Platinum DJ, producer and live instrumentalist Timmy Trumpet, internationally acclaimed didgeridoo player and composer William Barton and a 40-piece Orchestra will complete the grand final entertainment which will also include a spectacular light show. NRL chief executive Andrew Abdo said the week-long celebration was a fitting way to thank Queensland for supporting the Telstra Premiership in 2021 after 13 teams relocated to the Sunshine State in July

**Generated:** ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

Metadata: length=64, termination=max_new_tokens, repetition=0.984375, invalid_decode=False

### 46. `lora_step_0200` / `heldout_kyrgyz_00`

**Prompt:** Министрлер кабинетинин төрагасынын орун басары Эдил Байсалов Фейсбук барагында «Бириккен демократиялык кыймылынын» билдир

**Reference:** үүсүнө жооп берди. Кыймыл 13-апрелде Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген болчу. «Оппозиция деп бул досторубузду атоо туура эмес. Равшан Бабыр уулу 2,5 миң добуш алган президенттик шайлоодо. Клара айымдыкы дурусурак: 1% — 14 миң добуш. Анан кайдагы оппозиция боло алышмак эле? Активист дегилечи сураныч. Ошол туура болот!» — деп жазды Байсалов. Байсалов кыймылдын мүчөлөрүн оппозиция деп эсептебесин билдирди. «Уят эле бирок. Саясатчы менен артист сезиши керек сахнадан качан түшүштү. Эптеп эле чырлап, кыпчылып, өрдөк жокто чулдук бий болуу аракети. Оппозиция болот, азыр деле бар. Бирок бул активисттер эмес. Бузуку деп айтпайын. Бирок Кумтөрдүн 100% элибизге кайтарылышына БУУ баш болуп, дүйнөлүк коомчулук суктанып, алкыш айтып атышса, куттуктап, бу

**Generated:** ди. Байсалов өзүн өтүүнүн өзүн өтүүнүн өзүн өтүүнүн

Metadata: length=64, termination=max_new_tokens, repetition=0.781250, invalid_decode=False

### 47. `lora_step_0200` / `heldout_kyrgyz_01`

**Prompt:** лар тескери каршы чыгып аткандары бузукулук эмей эмне», — дейт чиновник. Байсаловдун билдирүүсүнө «Бириккен дем

**Reference:** ократиялык кыймылдын» мүчөсү, саясатчы Клара Сооронкулова дагы жооп кайтарды. «Байсаловду сөгө албаганыма эч качан өкүнгөн эмесмин. Кайда, кайсы кеменин артынан “жибереримди” билбей турам. Байкуш. Премьер-министрликке добуш берүү менен шайланса бир жөн эле», — деп жазды Сооронкулова Фейсбук барагында. Оппозиционерлердин билдирүүсү «Бириккен демократиялык кыймыл» 13-апрелде «Кумтөр: коррупциялык жана экологиялык тобокелчиликтер» деген темада өткөн жыйында Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген. Саясатчылар Кумтөрдү улутташтыруунун зарылдыгы тууралуу кайрылуу даярдашканын билдиришкен. Алардын пикиринде, канадалык компания менен келишим түзүү — Кыргызстан үчүн утулуш болот, анткени өлкө «Центеррадагы» акцияларынан баш тартып, экологиялык

**Generated:** илет» жана «Бириккен демилет» жана «Бириккен демилет» жана «Бириккен демилет» жана «Бириккен демилет» жана «Бириккен демилет» ж

Metadata: length=64, termination=max_new_tokens, repetition=0.812500, invalid_decode=False

### 48. `lora_step_0200` / `heldout_kyrgyz_02`

**Prompt:**  чыгымдарды өз мойнуна алып жатат. «Бириккен демократиялык кыймыл» — бул кыргыз оппозициясынын жаңы түзү

**Reference:** мү, кыймылдын түзүлгөнү 31-мартта белгилүү болгон. Анын курамында Равшан Жээнбеков, Клара Сооронкулова, Бектур Асанов, Мамбетжунус Абылов, Кеңешбек Дүйшөбаев өңдүү саясатчылар, ошондой эле укук коргоочулар жана активисттер бар. Кыймылдын координатору — Мамбетжунус Абылов.<|end_of_text|>Макензи, Маккензи – Канаданын түндүк-батышындагы дарыя. Чоң Эрксиз көлүнөн башталып, Макензи ойдуңу аркылуу агат да Түндүк Муз океандын Бофорт деңизине куят. Өзүнүн Узундугу 1770 кмдей, Пис-Ривер менен (Финли дарыясынын башынан) 4250 км, алабынын аянты (Чоң Эрксиз көлүнүн алабына кирген Эрксиз, Пис-Ривер жана Атабаска дарыясынын системасы менен) 1804 миң км2. Ири куймалары: Лиард, Арктик-Ред-Ривер, Пил (сол), Чоң Аюулуу (оң). Жээги өтө саздак, карагай токою менен

**Generated:** мдүн биринчи күнүн өтүүнүн өз айында өтүүнүн өз айында өтү

Metadata: length=64, termination=max_new_tokens, repetition=0.703125, invalid_decode=False

### 49. `lora_step_0200` / `heldout_kyrgyz_03`

**Prompt:**  капталган. Кар, жамгыр сууларынан курулат. Жаз жана жай мезгилдеринде ташкындайт. Орточо чыгымы 14 миң м3/сек. Октябр

**Reference:** дан июнга чейин тоңот. Кеме жолунун узундугу 2200 км. Боюнда Форт-Провиденс, Форт-Норман, Норман-Уэлс, Аклавик, Инувик калктуу пункттары жайгашкан. Дарыя, европалыктардын ичинен алгачкы жолу (1789) ал аркылуу сүзүп, Түндүк Муз океанга чыккан шотланддык көпөс, саякатчы – А. Макензинин урматынан аталган. Колдонулган адабияттар[түзөтүү | булагын түзөтүү] “Кыргызстан” улуттук энциклопедиясы: 5-том. Башкы редактору Асанов Ү. А. К 97. Б.: Мамлекеттик тил жана энциклопедия борбору, 2014. илл. ISBN 978 9967-14-111-7 "https://ky.wikipedia.org/w/index.php?title=Макензи&oldid=319180" булагынан алынды Категория: Канада Дарыялар Жашыруун категория: Pages using ISBN magic links Бул барак соңку жолу 5 май 2018, саат 12:39 өзгөртүлгөн. Текст Creative Commons Attribution-ShareAlike лицензиясынын алкагында жеткиликтүү; кошумча шарттар колдонулушу мүмкүн. Кеңири маалымат ү

**Generated:** да 1 миң м3/сек. Бирок 1 миң м3/сек. Бирок 1 миң м3/сек. Бирок 1 миң м3/сек. Бирок 1 миң м3

Metadata: length=64, termination=max_new_tokens, repetition=0.765625, invalid_decode=False

### 50. `lora_step_0200` / `heldout_kyrgyz_04`

**Prompt:** чүн Колдонуу шарттарын караңыз.<|end_of_text|>“Ысык-Көл” эл аралык аэропорту бүгүн, 20-ноябрда, 100 000-жүрг

**Reference:** үнчүсүн тейледи. Юбилейлик жүргүнчү 2021-жылдын 20-ноябрында “Аэрофлот” авиакомпаниясынын каттамы менен Москва шаарына учуп жаткан Жумабаева Чолпон болду. Бул тууралуу «Манас» эл аралык аэропортунун басма сөз кызматы билдирди. Бул — тарыхый окуя , себеби «Ысык-Көл» аэропорту алгачкы жолу 100 миң кишилик чектен ашты. Тактап айтсак, аэропорт 2021 — жылы 1000ден ашык рейсти тейлесе, бүгүн 100 000-жүргүнчүсүн тейледи. «Аймактардагы аэропортторубуз да өнүгүп, 100 миң кишилик чекке жеткенине кубанып турабыз Бул биз үчүн жакшы көрсөткүч. Биздин жүргүнчүбүздү чын дилимден куттуктайм” деп белгиледи «Манас» эл аралык аэропортунун башкармалыгынын төрагасы Бакыт Шаршеев. Өз кезегинде, «Ысык-Көл» эл аралык аэропортунун башчысы Алмаз Абдрахманов жүргү

**Generated:** үзүн. Бирок, өзүн өтүү, өзүн өтүү, өзүн өтүү, өз�

Metadata: length=64, termination=max_new_tokens, repetition=0.796875, invalid_decode=True

### 51. `lora_step_0200` / `kyrgyz_flores_00`

**Prompt:** Ал мурун "диабет оорусу менен ооруган, ал эми азыр оорудан айыккан 4 айлык чычкан бар деп кошумчалады".<|end_of_text|>Канаданын Диабет Ас

**Reference:** социациясынын илимий жана клиникалык бөлүмүнүн төрагасы жана Жаңы Шотландиянын Галифекс шаарындагы Далхаузи университетинин медицина илимдеринин профессору, доктор Эхуд Ур бул изилдөөнүн башталышы гана экенин эскертти.<|end_of_text|>Ал диабетти айыктыруу мүмкүнчүлүгүнөн айрым эксперттердей эле күмөн санайт жана бул натыйжалардын 1-типтеги диабет менен ооруган адамдарга тиешеси жок экенин белгилейт.<|end_of_text|>Дүйшөмбү күнү Швед академиясынын адабият боюнча Нобель комитетинин туруктуу катчысы Сара Даниус Швециядагы Sveriges Radio программасында 2016-жылдын адабияты боюнча Нобель сыйлыгын алуу жөнүндө Боб Дилан менен түздөн-түз байланыша албаган комитет аны менен байланышуу аракетин токтотконун жалпыга ачык билдирди.<|end_of_text|>"Учурда эч нерсе кылбай эле турабыз. Мен анын эң жакын кызматташына чалып, электрондук каттарды жөнөттү

**Generated:** оциациясынын басма байланышына өтүүнүн өтүүнүн өтүүнүн өтүүнүн �

Metadata: length=64, termination=max_new_tokens, repetition=0.703125, invalid_decode=True

### 52. `lora_step_0200` / `kyrgyz_flores_01`

**Prompt:** п, абдан жагымдуу жоопторду алдым. Бул албетте, азырынча жетиштүү", — деди Даниус.<|end_of_text|>Мындан мурда Ring компаниясынын башкы

**Reference:**  директору Жейми Симиноф компания анын гаражындагы дүкөндөн эшиктин шыңгыраганы угулбаганда иштей баштаганын белгилеген.<|end_of_text|>Ал WiFi менен иштеген эшик коңгуроосун жасаганын айтты.<|end_of_text|>Симиноффтун айтымында, ал 2013-жылы Shark Tank шоусунда чечим чыгарган топтор стартапты каржылоодон баш тарткан учурда чагылдырылгандан кийин, сатуулардын саны жогорулаган.<|end_of_text|>Симинофф 2017-жылдын аягында QVC соода телеканалына чыкты.<|end_of_text|>Ринг атаандаш коопсуздук компаниясы болгон ADT корпорациясы менен болгон соттук процессти жөнгө салган.<|end_of_text|>Бир эксперименталдык вакцина Эбола вирусунан болгон өлүмдү азайта алчудай, бирок бүгүнкү күнгө чейин инфекцияны дарылоого бир дагы дары ылайык көрсөтүлө элек.<|end_of_text|>ZMapp, антидене коктейли, бул жаатта өзүн көптөн үмүткөр кылган, бирок расмий изилдөөлөр көрсөткөндөй анын өлүм

**Generated:**  байланышында өтүү жана өтүү жана өтүү жана өтүү жана өтүү жана өтү�

Metadata: length=64, termination=max_new_tokens, repetition=0.765625, invalid_decode=True

### 53. `lora_step_0200` / `kyrgyz_flores_02`

**Prompt:** дү алдын алуудагы пайдасы аз.<|end_of_text|>PALM сыноосунда, ZMapp көзөмөлдөө катары кызмат кылган, атап айтканда о

**Reference:** кумуштуулар аны баштапкы план катары колдонушуп, ага үч башка дарылоону салыштырышкан.<|end_of_text|>USA Gymnastics АКШнын Олимпиадалык комитетинин катын колдойт жана бардык спортчуларыбыз үчүн коопсуз чөйрөнү тартуулоого олимпиада үй-бүлөсүнүн бардык муктаждыктарынын милдеттенмесин өзүнө алат.<|end_of_text|>Американын олимпиада комитети (USOC) тастыктамадан баш тарткандын ордуна, уюмубуздун ичинде олуттуу өзгөрүүлөрдү жасоо аркылуу клуб менен спортсмендерибиздин кызыкчылыктарын тейлей алабыз деген билдирүүсүнө макулбуз.<|end_of_text|>Америка Гимнастикасы Ларри Насардан аман калгандар көп убакыт бою байкалбаганын жана пропорцияны кыянаттык менен пайдаланууну көрсөтүшү мүмкүн болгон көз карандысыз иликтөөнү колдойт жана керектүү өзгөртүүлөрдү киргизиши мүмкүн.<|end_of_text|>АКШ Гимнастикасы жана Олимпиадалы

**Generated:** шондой эле күнүн өтүүнүн өтүүнүн өтүүнүн өтүүнүн өт

Metadata: length=64, termination=max_new_tokens, repetition=0.796875, invalid_decode=False

### 54. `lora_step_0200` / `kyrgyz_flores_03`

**Prompt:** к Комитети бир эле максатты көздөшөт — гимнастиканы жана спорттун башка түрлөрүн спортчулар үчүн мүм

**Reference:** күн болушунча коопсуз кылып, кыялдарынын коопсуз, позитивдүү жана укуктуу чөйрөдө ишке ашуусун камсыз кылуу.<|end_of_text|>1960-жж. Бржезинский Джон Кеннединин кеңешчиси болуп, кийинчерээк Линдон Джонсондун кол алдында иштеген.<|end_of_text|>1976-ж. шайлоодо Картердин тышкы саясатты алып баруу боюнча кеңешчиси, кийинчерээк 1977-1981-жж. Генри Киссинджердин ордуна Улуттук коопсуздук боюнча кеңешчи (NSA) болгон.<|end_of_text|>Улуттук башкаруу агенттиги катары ал Картерге төмөнкүдөй жалпы көйгөйлөрдү дипломаттык түрдө чечүүгө жардам берген: 1978-жылдагы Кэмп-Дэвид макулдашуусу, 1970-жылдын аягында америка-кытай мамилелерин калыбына келтирүү, 1979-жылы Иранды кризиске алып келген Иран төңкөрүшү, 1979-жылы Советтер союзунун Афганистанга кол салышы.<|end_of_text|>Эмма Стоун жана Раян Гослинг катышкан фильм негизги категориялардын ба

**Generated:** күн. Бирок, бир эле максатты көздөшөтүүнүн биринчи жана сизди биринчи жана сизди биринчи жана сизди бир

Metadata: length=64, termination=max_new_tokens, repetition=0.546875, invalid_decode=False

### 55. `lora_step_0200` / `kyrgyz_flores_04`

**Prompt:** арында номинация алган.<|end_of_text|>Гослинг жана Стоун эң мыкты актер жана эң мыкты актриса номинацияларын алышты.<|end_of_text|>Башка номинацияларга эң

**Reference:**  мыкты сүрөт, режиссер, оператордук иш, костюмдар дизайны, тасмаларды монтаждоо, түпнуска саундтрек, өндүрүш дизайны, үн монтаждоо, үндөрдү аралаштыруу жана өзгөчө сценарий кирет.<|end_of_text|>Кинодогу "Угуу" (Кыялданган акылы жоктор) жана "Жылдыздардын шаары" деген эки ыр мыкты оригиналдуу ыр номинациясын алышты. Башка студиялардан өзгөчөлөнүп, Lionsgate студиясы 26 номинацияга ээ болду.<|end_of_text|>АКШнын президенти Дональд Трамп кечки жекшембиде пресс-катчы аркылуу жасаган билдирүүсүндө, АКШ аскерлери Сириядан чыгарын билдирди.<|end_of_text|>Бул жаңылык Трамп Түркиянын президенти Режеп Тайип Эрдоган менен телефондо жолугушкандан кийин билдирилди.<|end_of_text|>Билдирүүдө айтылгандай Европа мамлекеттери мекенине кайтарууну четке кагып, колго түшкөн ИГИЛ согушкерлерин кайтарууну Түркия колго алганын билдирди.<|end_of_text|>Бул айрым дин

**Generated:**  айырмалган актерлер: «Кыргызстандын өзүңүз» номинацияларын алышты. Бул тууралуу өзүңүз актерлердин

Metadata: length=64, termination=max_new_tokens, repetition=0.359375, invalid_decode=False

### 56. `lora_step_0200` / `english_ood_00`

**Prompt:** Barovier&Toso is a Venetian company with a vocation and culture that is international, creating solutions of decorative lighting in Murano glass, all characterised by a personal and unique style. ​ Lighting solutions destined not only to furnish ambiances, but also to create emotions, transmit positivity, making each room

**Reference:**  unique. ​ Traditions and the future Murano glass crafting tradition, of which we have always been absolute protagonists, is for us the starting point from which to propose ideas, solutions, innovative and original projects that have allowed us to hand down values through the century old history of the founders’ family.<|end_of_text|>Beckton locksmith services are offered by us, locks in the City. A true 24 hour emergency service, we have the solution anytime Day or Night. When you call our Beckton Locksmith service we speak with an actual locksmith. Daniel or myself Paul. With a combined 25 years experience and knowledge in all kinds of locks and hardware. We will be more then happy to attend your property for a free survey Beckton Locksmith services are used by domestic and commercial customers. We have customers within the retail parks that we carry out many typical locksmithing duties. We also have a retail shop for customers to come to us if they wish to. Where you can purchase anything from novelty keys through to Locks and associated hardware. If you need us to attend to fit the item we would be more then willing as we are the local Beckton Locksmith. Call Locks in the City today on 07921 000672 for a fast and friendly response. Our locksmiths are fully insured and are members of the Master Locksmiths Association (MLA). The team has been inspected by the MLA. We are fully qualified & fully insured. Consequently, you can have complete peace of mind when your Locks in the City representative arrives at your address. You know that they will be qualified to do the work. We use non destructive entry methods where ever possible in emergency lock out situations.<|end_of_text|>match preview: The Exiles head to rugby lane and sutton & epsom, for the first time - in round 6 - London Welsh Rugby Club Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh Home Teams Men’s Men’s Rugby Performance Squad Women & Girls Women

**Generated:**  a unique space. The company was founded in 1990 by the brothers Vittorio and Giorgio Barovier, who have always been passionate about the art of glass. The company is named after the two founders, who have always been guided by the desire to create unique and exclusive products. The company has

Metadata: length=64, termination=max_new_tokens, repetition=0.265625, invalid_decode=False

### 57. `lora_step_0200` / `english_ood_01`

**Prompt:** ’s Rugby Women’s Squad Girls Youth & Minis Youth Rugby Mini Rugby Fixtures Latest News News & Media News Videos Gallery Podcast Membership Supporters club About Us Join Us Travel And Events The 400 Club Careers Hub Hub News The Hub The Idea Success Stories The Mentoring Experience Business Partners Contact The Hub Events & Hospitality LWRFC

**Reference:**  Choir Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh match preview: The Exiles head to rugby lane and sutton & epsom, for the first time – in round 6 News October 9, 2022 Following the disappointment of Round 5 and yet another top 4 side that Welsh will feel they should have beaten, Round 6 is yet another first time fixture for the London Welsh faithful. The Exiles fourth away trip of the season takes them to Ewell in Sutton and a side that Welsh meet for the first time in their history (something to which we have become accustom to since project reset). Sutton & Epsom has a similar longevity to that of London Welsh, playing their first matches in the 1883/84 season – the season prior to the founding of London Welsh. In recent seasons Sutton have found the going tough after promotion in 18/19 season from Level 5 to level 4 and National 2 South. Sutton are one of the sides to count themselves unlucky from the RFU calculations resulting from the 80% played season of covid riven 19/20. Sutton finished the played season 3rd from bottom and were relegated with no opportunity to complete the season and try to reel in 4th from bottom Westcliff. That said, Sutton were 14 points adrift of safety at that stage. In the first full season post Covid, Sutton finished 11th in South Premier last season (4th from bottom) and comfortably clear of Brighton in 12th. Although early in the season both sides will see this Round 6 fixture as crucial: Sutton currently sit just one place above Welsh in the league (9th) and with an identical tally of won 1 lost 4 (Sutton with 2 more bonus points to their name). Welsh will feel that they could and perhaps should have won all 4 of their previous fixtures, having taken 3 of the current top

**Generated:**  LWRFC 2023-24 Season 2023-24 Season 2023-24 Season 2023-24 Season 2023-24 Season 2023-24 Season 2023-24 Season 2023-24 Season 2023-24 Season 2023-24 Season

Metadata: length=64, termination=max_new_tokens, repetition=0.859375, invalid_decode=False

### 58. `lora_step_0200` / `english_ood_02`

**Prompt:** 4 sides all the way. That all said, Welsh need league points in order to get their season properly underway. It is well accepted that Welsh have been victims of an almost unprecedented injury crisis in the early rounds, an issue that is only now seeing signs of abating. In the latest injury and availability news: Top

**Reference:**  try scorer Sion Cowdy remains on the side-lines; he is on track to return in Round 7 and is with the Ireland 7s squad in Spain this weekend. There will be a fitness test this week for Loosehead Griff Whitson after he too missed Round 6 following injury in Round 5. Will Ponty at second row suffered a concussion in Round 5 and will be undergo HIA protocol assessment this week as he clears the mandatory 3 weeks on the side-lines. Former Premiership star Ben West is almost ready for his first game of the season but Round 6 will come just that bit too early. In positive news: Matt Pollen returns at centre for the first time since Round 2, Wales u18 man Rhys Howells also returns in the centre for potentially his 53rd cap and following his wedding last weekend. Hywel Williams also returns at to contest the starting spot at 9 having been unavailable for round 5. Tom Williams played 80 minutes for the VETS (as their u35 player) against Wasps on Friday last as he continues his 6 month return to fitness, he may be in contention for a first game of the season. James Smith (broken shoulder) and Jack Anderson (dislocated knee) remain the long term absentees. Welsh could jump as high as 7th win a win in Round 6 and will want to continue their excellent form and growth that we have seen since round 4. Director of Men’s Rugby Cai Griffiths said of the game ” We have another big week in training this week with the squad returning to some form of normality and continuity, given the injuries that we have had. We expect this Round 6 match to be another tough contest with Sutton & Epsom coming close in many of their games so far this season. We have certainly felt like we are not far off where we need to be in the past few weeks and with a more settled team the squad will be keen to prove some points’. No doubt eyes will be on some of the big performers from the last few rounds: Samoan Kiwi Tololima Saavainaea has be excellent in his first three games and grabbed his

**Generated:**  scorer Rhys Bennett is out for the season after suffering a broken leg in the 2nd round. The Welsh manager has also been forced to make a number of changes to his squad, with the likes of James McArthur, Ryan Williams, and Ben Davies all out for the season. The Welsh manager has also been

Metadata: length=64, termination=max_new_tokens, repetition=0.281250, invalid_decode=False

### 59. `lora_step_0200` / `english_ood_03`

**Prompt:**  first try in Round 5 against Havant. Ben Bodinham has been excellent having moved to 8 from the second row, following the loss of Jack Anderson at 8 for the rest of the season. Supporters and the squad will welcome the return of Matt Pollen at centre and hopes will be high that he

**Reference:**  makes a big impact on first game back since Round 2. Top points scorer Osian McAvoy has been excellent following a switch this season from 10 to fullback and his talent will not stand alone in a talented back line. Kick off will be 3pm at Sutton & Epsom RFC, Cuddington Court, Rugby Lane. Details will be released this week on the supporters transport to match. We hope to see the large travelling exile army with us as always. London Welsh’s next home game will come in Round 7 on 22 October when Bracknell RFC visit Old Deer Park. #COYW share: Post navigation Previous Post Match report: bp win despite the odds for LWWRFC Next Post Lasyl and Location:Live sponsor London Welsh Colts Academy As they Take on Monmouth school You May Also Like News November 11, 2022 the sad loss of Don mcmanus It is with great sadness that the London Welsh RFC community this week has learnt of the death in… News December 9, 2020 The Boys are back in town London Welsh emerge from Lockdown 2.0 with the go-ahead to resume competitive contact rugby! With some tasty fixtures already…<|end_of_text|>He'll be out for eight weeks which means he will still be available for the All Blacks World Cup campaign in France in September. Moody's the fourth prop the Crusaders have lost during their title defence. The Super Rugby Pacific side has already lost Fletcher Newell, George Bower and Finlay Brewis to injuries. The Crusaders hope to add another prop to their roster in the near future. 34 year old Moody missed much of last season with an ACL injury. He played at both the 2015 and 2019 World Cups. Meanwhile, Blues coach Leon Macdonald has confirmed first-five Beauden Barrett will miss Saturday's game against the Hurricanes. Barrett has a heel injury picked up in the last round against the Reds in Brisbane. "He's on his All Blacks rest this week which coincides with his stitches in his foot, it's hard to know for sure but we're pretty confident he's going to be okay [for next week]." Macdonald says Barrett needed to

**Generated:**  can help the team to a better finish than last season. The squad is looking forward to the new season and the challenge ahead. The squad is as follows: 1. Ben Bodinham 2. Jack Anderson 3. Matt Pollen 4. Ben Hargreaves 5. Ben Harg

Metadata: length=64, termination=max_new_tokens, repetition=0.328125, invalid_decode=False

### 60. `lora_step_0200` / `english_ood_04`

**Prompt:**  take his mandatory All Blacks rest this round or next but his injury forced their hand. The fourth placed Blues meet the fifth placed Hurricanes at Eden Park in a battle for fourth spot or higher on the Super Rugby Pacific table with a home playoff game on the line.<|end_of_text|>A galaxy of stars will headline the NRL grand final entertainment

**Reference:**  while Brisbane will pay tribute to rugby league royalty in the lead-up to the game with King George Square renamed King Wally Square. Grand Final Week officially kicked off on Sunday evening with Brisbane City Hall, King George Tower and the Hilton lit up with iconic images from the season, tributes to Penrith and South Sydney, the game’s fans and a thank you to the Queensland community for their hospitality. King George Square in the Brisbane CBD is being renamed King Wally Square in honour of Queensland Immortal Wally Lewis as part of a two-day NRL Fan Fest on Thursday and Friday, culminating with the appearance of both teams on Friday afternoon. "I think the name choice is fantastic," Lewis said with a laugh on Sunday night. "A lot of people have been asking me if I think it's a little bit of a giggle fest. Yeah, I do but after King George passed away I think it was 83 years ago, it was put in his honour. I don't have any plans to be passing away soon. "To have an honour like this certainly is wonderful but I'm fully aware it's only going to be going on whilst the rugby league grand final is being put in the planners and it's going to be going back to King George Square immediately after the game is finished." Lewis said he was extremely proud of the people of Queensland for offering their support to the NRL players in recent months during difficult circumstances." Next Sunday's pre-game entertainment at Suncorp Stadium will have a Queensland theme with home-grown star Kate Miller-Heidke to sing the national anthem and the award-winning singer-songwriter will also lead the entertainment with Ian Moss, respected across the globe as one of Australia’s most iconic musicians. Globe-trotting DJs and producers The Stafford Brothers, multi-Platinum DJ, producer and live instrumentalist Timmy Trumpet, internationally acclaimed didgeridoo player and composer William Barton and a 40-piece Orchestra will complete the grand final entertainment which will also include a spectacular light show. NRL chief executive Andrew Abdo said the week-long celebration was a fitting way to thank Queensland for supporting the Telstra Premiership in 2021 after 13 teams relocated to the Sunshine State in July

**Generated:**  at the Sydney Cricket Ground on Saturday night. The NRL grand final entertainment will include a 10-minute performance by the legendary Australian rock band AC/DC, a 15-minute performance by the legendary Australian rock band Cold Chisel, a 10-minute performance by the legendary Australian rock band INXS, a 10-minute

Metadata: length=64, termination=max_new_tokens, repetition=0.437500, invalid_decode=False

## Human Interpretation

Interpretation is based on all raw outputs above, not aggregate metrics alone.

- Frozen base: heldout and FLORES Kyrgyz outputs repeatedly copy fragments or enter character/word loops; English is often topical but includes copied lists and repeated continuations.
- Axis step 200: Kyrgyz remains repetitive or malformed. English is mostly topical, but the raw outputs still contain copied continuations and no EOS termination.
- Axis step 2000: several heldout Kyrgyz outputs become more locally grammatical, but FLORES remains inconsistent and English collapses to comma punctuation in all five examples. This is not a coherent cross-source capability result.
- LoRA step 200: Kyrgyz and FLORES remain repetitive or malformed in the same way as the earlier audit. English retains some topical continuation but does not remove copying or looping.
- Because every condition used `use_cache=False`, the earlier generation degeneration remains after authoritative decoding. The cache discrepancy is not sufficient to explain the failure.

The first Phase 2 launch failed before generation because the Axis hook received BF16 activations and FP32 deltas without the established autocast context: `RuntimeError: expected mat1 and mat2 to have the same dtype`. The corrected launch reused the prior BF16 autocast path and produced all 60 records in `92.504s`.
