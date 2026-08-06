# LoRA Exposure Curve

This is a diagnostic LoRA control only; it is not part of Grafting or Axis ARW. All generation uses `use_cache=False`.

## Metrics

```json
{
  "model": "/workspace/model/real_SmolLM3-3B",
  "config": {
    "seed": 42,
    "batch": 1,
    "max_len": 512,
    "learning_rate": 0.0002,
    "weight_decay": 0.01,
    "optimizer": "AdamW",
    "autocast_dtype": "torch.bfloat16",
    "target_modules": [
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    "rank": 32,
    "alpha": 64,
    "dropout": 0.0,
    "bias": "none",
    "use_cache_training": false
  },
  "paths": {
    "train": "/workspace/kyrgyz_train.bin",
    "heldout": "/workspace/kyrgyz_heldout.bin",
    "flores": "/workspace/kyrgyz_flores.bin",
    "ood": "/workspace/kyrgyz_english_ood.bin"
  },
  "trainable_parameters": 45121536,
  "checkpoint_steps": [
    {
      "step": 200,
      "actual_supervised_tokens_seen": 102200,
      "training_loss": 2.6577353477478027,
      "evaluation": {
        "heldout_kyrgyz": {
          "ce": 2.4837931018986112,
          "ppl": 11.986644862499938
        },
        "kyrgyz_flores": {
          "ce": 2.6211410646719746,
          "ppl": 13.751405879559403
        },
        "english_ood": {
          "ce": 2.6252554316809964,
          "ppl": 13.808100762037322
        }
      },
      "wall_time_s": 483.7159407469444,
      "checkpoint_path": "/workspace/GLT/experiments/kyrgyz_lora_exposure/checkpoints/step_0200",
      "artifact_hashes": {
        "files": {
          "README.md": "4e72ac623b80ff3f7a10bdff4fc2a47f65127c899e120f733835b1ec5e413b4d",
          "adapter_config.json": "0cc8ca9b546edac1ecfefa331a7f231410f1f25c55bb897be0ee788a0d5e1081",
          "adapter_model.safetensors": "6ac2ca1421fe5c8f7b15724933d62df955f162cd2491c82b22e6b8ef7d1f006d",
          "optimizer.pt": "ab11bbb72fd467af4f918d8a6308b4948bbae4b408cd05d9e7de13512fc09737"
        },
        "manifest": "4643eff9cfc4f8773c74981153364031c5f886d93df4368692891e612cf4bfb5"
      },
      "generation_records_added": 6
    },
    {
      "step": 1000,
      "actual_supervised_tokens_seen": 511000,
      "training_loss": 1.2746273279190063,
      "evaluation": {
        "heldout_kyrgyz": {
          "ce": 2.0943128246150606,
          "ppl": 8.119859285204035
        },
        "kyrgyz_flores": {
          "ce": 2.280035794551125,
          "ppl": 9.777030367678918
        },
        "english_ood": {
          "ce": 2.6483361178648215,
          "ppl": 14.130507575225579
        }
      },
      "wall_time_s": 1213.0219465112314,
      "checkpoint_path": "/workspace/GLT/experiments/kyrgyz_lora_exposure/checkpoints/step_1000",
      "artifact_hashes": {
        "files": {
          "README.md": "4e72ac623b80ff3f7a10bdff4fc2a47f65127c899e120f733835b1ec5e413b4d",
          "adapter_config.json": "0cc8ca9b546edac1ecfefa331a7f231410f1f25c55bb897be0ee788a0d5e1081",
          "adapter_model.safetensors": "a66a09b8ac638e636c366eee4409877bdc004624ed12d2f68e21fdf8e847f72f",
          "optimizer.pt": "06eabafa8ed10263932bcc235daedd703793bb850b4ececedd98e6c0a4765e46"
        },
        "manifest": "3f585828366b3eb325db5f33f68c5c502616ac1a90b36b0860ef6097147ec9ed"
      },
      "generation_records_added": 6
    },
    {
      "step": 2000,
      "actual_supervised_tokens_seen": 1022000,
      "training_loss": 1.4307312965393066,
      "evaluation": {
        "heldout_kyrgyz": {
          "ce": 1.932014999091042,
          "ppl": 6.9034065943230205
        },
        "kyrgyz_flores": {
          "ce": 2.134588153367843,
          "ppl": 8.453564217873424
        },
        "english_ood": {
          "ce": 2.6731794653806666,
          "ppl": 14.4859536391889
        }
      },
      "wall_time_s": 2024.7604182050563,
      "checkpoint_path": "/workspace/GLT/experiments/kyrgyz_lora_exposure/checkpoints/step_2000",
      "artifact_hashes": {
        "files": {
          "README.md": "4e72ac623b80ff3f7a10bdff4fc2a47f65127c899e120f733835b1ec5e413b4d",
          "adapter_config.json": "0cc8ca9b546edac1ecfefa331a7f231410f1f25c55bb897be0ee788a0d5e1081",
          "adapter_model.safetensors": "b2623b20c3a4d96934b9572c5c75dc8940c15512d483a829dd08e80b44973b7e",
          "optimizer.pt": "c47d1ee15cd727c9a492a1d5d1409e1741dc7dc70971943d132a16bb86e7773b"
        },
        "manifest": "b5761d28434248e00913df38831e7778d3d8d55ffceceda3fe141896766e8068"
      },
      "generation_records_added": 6
    },
    {
      "step": 5000,
      "actual_supervised_tokens_seen": 2555000,
      "training_loss": 1.1728620529174805,
      "evaluation": {
        "heldout_kyrgyz": {
          "ce": 1.8097186881735368,
          "ppl": 6.108728732859506
        },
        "kyrgyz_flores": {
          "ce": 2.0906861484340475,
          "ppl": 8.09046451980753
        },
        "english_ood": {
          "ce": 2.8297273229712845,
          "ppl": 16.940840816546817
        }
      },
      "wall_time_s": 3659.865242946893,
      "checkpoint_path": "/workspace/GLT/experiments/kyrgyz_lora_exposure/checkpoints/step_5000",
      "artifact_hashes": {
        "files": {
          "README.md": "4e72ac623b80ff3f7a10bdff4fc2a47f65127c899e120f733835b1ec5e413b4d",
          "adapter_config.json": "0cc8ca9b546edac1ecfefa331a7f231410f1f25c55bb897be0ee788a0d5e1081",
          "adapter_model.safetensors": "c67de2de731c3b9a2062ad164f59702cb18ca321c576125d24f8faee41a75393",
          "optimizer.pt": "346b1fabb6c74e408732ad7c2be14dac316b7a601ded7b8f33b77f61bd08d9c1"
        },
        "manifest": "ea2769e182236014999824990e2bd26d50328f7f2d52a589668ac62d4bae81ad"
      },
      "generation_records_added": 6
    }
  ],
  "loss_at_steps": {
    "200": 2.6577353477478027,
    "1000": 1.2746273279190063,
    "2000": 1.4307312965393066,
    "5000": 1.1728620529174805
  },
  "peak_vram_allocated_mb": 12761.06640625,
  "peak_vram_reserved_mb": 18742.0,
  "wall_time_s": 3659.8655373840593,
  "generation_records": 24,
  "final_training_loss": 1.1728620529174805
}
```

## Complete Generation Samples

### Step 200

#### `heldout_kyrgyz_00` (heldout_kyrgyz)

**Prompt:** Министрлер кабинетинин төрагасынын орун басары Эдил Байсалов Фейсбук барагында «Бириккен демократиялык кыймылынын» билдир

**Reference:** үүсүнө жооп берди. Кыймыл 13-апрелде Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген болчу. «Оппозиция деп бул досторубузду атоо туура эмес. Равшан Бабыр уулу 2,5 миң добуш алган президенттик шайлоодо. Клара айымдыкы дурусурак: 1% — 14 миң добуш. Анан кайдагы оппозиция боло алышмак эле? Активист дегилечи сураныч. Ошол туура болот!» — деп жазды Байсалов. Байсалов кыймылдын мүчөлөрүн оппозиция деп эсептебесин билдирди. «Уят эле бирок. Саясатчы менен артист сезиши керек сахнадан качан түшүштү. Эптеп эле чырлап, кыпчылып, өрдөк жокто чулдук бий болуу аракети. Оппозиция болот, азыр деле бар. Бирок бул активисттер эмес. Бузуку деп айтпайын. Бирок Кумтөрдүн 100% элибизге кайтарылышына БУУ баш болуп, дүйнөлүк коомчулук суктанып, алкыш айтып атышса, куттуктап, бу

**Generated:** ди. Байсалов өзүн өтүүнүн өтүүнүн өтүүнүн өтүүнүн ө

Metadata: length=64, termination=max_new_tokens, repetition=0.781250, invalid_decode=False

#### `heldout_kyrgyz_01` (heldout_kyrgyz)

**Prompt:** лар тескери каршы чыгып аткандары бузукулук эмей эмне», — дейт чиновник. Байсаловдун билдирүүсүнө «Бириккен дем

**Reference:** ократиялык кыймылдын» мүчөсү, саясатчы Клара Сооронкулова дагы жооп кайтарды. «Байсаловду сөгө албаганыма эч качан өкүнгөн эмесмин. Кайда, кайсы кеменин артынан “жибереримди” билбей турам. Байкуш. Премьер-министрликке добуш берүү менен шайланса бир жөн эле», — деп жазды Сооронкулова Фейсбук барагында. Оппозиционерлердин билдирүүсү «Бириккен демократиялык кыймыл» 13-апрелде «Кумтөр: коррупциялык жана экологиялык тобокелчиликтер» деген темада өткөн жыйында Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген. Саясатчылар Кумтөрдү улутташтыруунун зарылдыгы тууралуу кайрылуу даярдашканын билдиришкен. Алардын пикиринде, канадалык компания менен келишим түзүү — Кыргызстан үчүн утулуш болот, анткени өлкө «Центеррадагы» акцияларынан баш тартып, экологиялык

**Generated:** илет» жана «Бириккен демилет» жана «Бириккен демилет» жана «Бириккен демилет» жана «Бириккен демилет» жана «Бириккен демилет» ж

Metadata: length=64, termination=max_new_tokens, repetition=0.812500, invalid_decode=False

#### `heldout_kyrgyz_02` (heldout_kyrgyz)

**Prompt:**  чыгымдарды өз мойнуна алып жатат. «Бириккен демократиялык кыймыл» — бул кыргыз оппозициясынын жаңы түзү

**Reference:** мү, кыймылдын түзүлгөнү 31-мартта белгилүү болгон. Анын курамында Равшан Жээнбеков, Клара Сооронкулова, Бектур Асанов, Мамбетжунус Абылов, Кеңешбек Дүйшөбаев өңдүү саясатчылар, ошондой эле укук коргоочулар жана активисттер бар. Кыймылдын координатору — Мамбетжунус Абылов.<|end_of_text|>Макензи, Маккензи – Канаданын түндүк-батышындагы дарыя. Чоң Эрксиз көлүнөн башталып, Макензи ойдуңу аркылуу агат да Түндүк Муз океандын Бофорт деңизине куят. Өзүнүн Узундугу 1770 кмдей, Пис-Ривер менен (Финли дарыясынын башынан) 4250 км, алабынын аянты (Чоң Эрксиз көлүнүн алабына кирген Эрксиз, Пис-Ривер жана Атабаска дарыясынын системасы менен) 1804 миң км2. Ири куймалары: Лиард, Арктик-Ред-Ривер, Пил (сол), Чоң Аюулуу (оң). Жээги өтө саздак, карагай токою менен

**Generated:** мдүн биринчи күнүн өтүүнүн өтүүнүн өтүүнүн өтүүнүн

Metadata: length=64, termination=max_new_tokens, repetition=0.781250, invalid_decode=False

#### `english_ood_00` (english_ood)

**Prompt:** Barovier&Toso is a Venetian company with a vocation and culture that is international, creating solutions of decorative lighting in Murano glass, all characterised by a personal and unique style. ​ Lighting solutions destined not only to furnish ambiances, but also to create emotions, transmit positivity, making each room

**Reference:**  unique. ​ Traditions and the future Murano glass crafting tradition, of which we have always been absolute protagonists, is for us the starting point from which to propose ideas, solutions, innovative and original projects that have allowed us to hand down values through the century old history of the founders’ family.<|end_of_text|>Beckton locksmith services are offered by us, locks in the City. A true 24 hour emergency service, we have the solution anytime Day or Night. When you call our Beckton Locksmith service we speak with an actual locksmith. Daniel or myself Paul. With a combined 25 years experience and knowledge in all kinds of locks and hardware. We will be more then happy to attend your property for a free survey Beckton Locksmith services are used by domestic and commercial customers. We have customers within the retail parks that we carry out many typical locksmithing duties. We also have a retail shop for customers to come to us if they wish to. Where you can purchase anything from novelty keys through to Locks and associated hardware. If you need us to attend to fit the item we would be more then willing as we are the local Beckton Locksmith. Call Locks in the City today on 07921 000672 for a fast and friendly response. Our locksmiths are fully insured and are members of the Master Locksmiths Association (MLA). The team has been inspected by the MLA. We are fully qualified & fully insured. Consequently, you can have complete peace of mind when your Locks in the City representative arrives at your address. You know that they will be qualified to do the work. We use non destructive entry methods where ever possible in emergency lock out situations.<|end_of_text|>match preview: The Exiles head to rugby lane and sutton & epsom, for the first time - in round 6 - London Welsh Rugby Club Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh Home Teams Men’s Men’s Rugby Performance Squad Women & Girls Women

**Generated:**  a unique space. The company was founded in 1990 by the brothers Roberto and Alessandro Barovier, who have always been passionate about glass and light. The company is named after the two founders, Roberto and Alessandro Barovier, who are the sons of the famous Murano glassmaker, Roberto Bar

Metadata: length=64, termination=max_new_tokens, repetition=0.328125, invalid_decode=False

#### `english_ood_01` (english_ood)

**Prompt:** ’s Rugby Women’s Squad Girls Youth & Minis Youth Rugby Mini Rugby Fixtures Latest News News & Media News Videos Gallery Podcast Membership Supporters club About Us Join Us Travel And Events The 400 Club Careers Hub Hub News The Hub The Idea Success Stories The Mentoring Experience Business Partners Contact The Hub Events & Hospitality LWRFC

**Reference:**  Choir Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh match preview: The Exiles head to rugby lane and sutton & epsom, for the first time – in round 6 News October 9, 2022 Following the disappointment of Round 5 and yet another top 4 side that Welsh will feel they should have beaten, Round 6 is yet another first time fixture for the London Welsh faithful. The Exiles fourth away trip of the season takes them to Ewell in Sutton and a side that Welsh meet for the first time in their history (something to which we have become accustom to since project reset). Sutton & Epsom has a similar longevity to that of London Welsh, playing their first matches in the 1883/84 season – the season prior to the founding of London Welsh. In recent seasons Sutton have found the going tough after promotion in 18/19 season from Level 5 to level 4 and National 2 South. Sutton are one of the sides to count themselves unlucky from the RFU calculations resulting from the 80% played season of covid riven 19/20. Sutton finished the played season 3rd from bottom and were relegated with no opportunity to complete the season and try to reel in 4th from bottom Westcliff. That said, Sutton were 14 points adrift of safety at that stage. In the first full season post Covid, Sutton finished 11th in South Premier last season (4th from bottom) and comfortably clear of Brighton in 12th. Although early in the season both sides will see this Round 6 fixture as crucial: Sutton currently sit just one place above Welsh in the league (9th) and with an identical tally of won 1 lost 4 (Sutton with 2 more bonus points to their name). Welsh will feel that they could and perhaps should have won all 4 of their previous fixtures, having taken 3 of the current top

**Generated:**  LWRFC 2023-24 Season 2023-24 Season 2023-24 Season 2023-24 Season 2023-24 Season 2023-24 Season 2023-24 Season 2023-24 Season 2023-24 Season 2023-24 Season

Metadata: length=64, termination=max_new_tokens, repetition=0.859375, invalid_decode=False

#### `english_ood_02` (english_ood)

**Prompt:** 4 sides all the way. That all said, Welsh need league points in order to get their season properly underway. It is well accepted that Welsh have been victims of an almost unprecedented injury crisis in the early rounds, an issue that is only now seeing signs of abating. In the latest injury and availability news: Top

**Reference:**  try scorer Sion Cowdy remains on the side-lines; he is on track to return in Round 7 and is with the Ireland 7s squad in Spain this weekend. There will be a fitness test this week for Loosehead Griff Whitson after he too missed Round 6 following injury in Round 5. Will Ponty at second row suffered a concussion in Round 5 and will be undergo HIA protocol assessment this week as he clears the mandatory 3 weeks on the side-lines. Former Premiership star Ben West is almost ready for his first game of the season but Round 6 will come just that bit too early. In positive news: Matt Pollen returns at centre for the first time since Round 2, Wales u18 man Rhys Howells also returns in the centre for potentially his 53rd cap and following his wedding last weekend. Hywel Williams also returns at to contest the starting spot at 9 having been unavailable for round 5. Tom Williams played 80 minutes for the VETS (as their u35 player) against Wasps on Friday last as he continues his 6 month return to fitness, he may be in contention for a first game of the season. James Smith (broken shoulder) and Jack Anderson (dislocated knee) remain the long term absentees. Welsh could jump as high as 7th win a win in Round 6 and will want to continue their excellent form and growth that we have seen since round 4. Director of Men’s Rugby Cai Griffiths said of the game ” We have another big week in training this week with the squad returning to some form of normality and continuity, given the injuries that we have had. We expect this Round 6 match to be another tough contest with Sutton & Epsom coming close in many of their games so far this season. We have certainly felt like we are not far off where we need to be in the past few weeks and with a more settled team the squad will be keen to prove some points’. No doubt eyes will be on some of the big performers from the last few rounds: Samoan Kiwi Tololima Saavainaea has be excellent in his first three games and grabbed his

**Generated:**  scorer Rhys Bennett is out for the season after suffering a broken leg in the opening round. The 24-year-old was a key player for Welsh in the first round, scoring 10 goals in 5 games. Welsh have also lost the services of 2 other players in the opening round, with 2nd

Metadata: length=64, termination=max_new_tokens, repetition=0.312500, invalid_decode=False

### Step 1000

#### `heldout_kyrgyz_00` (heldout_kyrgyz)

**Prompt:** Министрлер кабинетинин төрагасынын орун басары Эдил Байсалов Фейсбук барагында «Бириккен демократиялык кыймылынын» билдир

**Reference:** үүсүнө жооп берди. Кыймыл 13-апрелде Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген болчу. «Оппозиция деп бул досторубузду атоо туура эмес. Равшан Бабыр уулу 2,5 миң добуш алган президенттик шайлоодо. Клара айымдыкы дурусурак: 1% — 14 миң добуш. Анан кайдагы оппозиция боло алышмак эле? Активист дегилечи сураныч. Ошол туура болот!» — деп жазды Байсалов. Байсалов кыймылдын мүчөлөрүн оппозиция деп эсептебесин билдирди. «Уят эле бирок. Саясатчы менен артист сезиши керек сахнадан качан түшүштү. Эптеп эле чырлап, кыпчылып, өрдөк жокто чулдук бий болуу аракети. Оппозиция болот, азыр деле бар. Бирок бул активисттер эмес. Бузуку деп айтпайын. Бирок Кумтөрдүн 100% элибизге кайтарылышына БУУ баш болуп, дүйнөлүк коомчулук суктанып, алкыш айтып атышса, куттуктап, бу

**Generated:** ди. «Бириккен демократиялык кыймылынын» өкүлдөрүнүн басма сөз кызматынан билдиришти. «Бирикк

Metadata: length=64, termination=max_new_tokens, repetition=0.390625, invalid_decode=False

#### `heldout_kyrgyz_01` (heldout_kyrgyz)

**Prompt:** лар тескери каршы чыгып аткандары бузукулук эмей эмне», — дейт чиновник. Байсаловдун билдирүүсүнө «Бириккен дем

**Reference:** ократиялык кыймылдын» мүчөсү, саясатчы Клара Сооронкулова дагы жооп кайтарды. «Байсаловду сөгө албаганыма эч качан өкүнгөн эмесмин. Кайда, кайсы кеменин артынан “жибереримди” билбей турам. Байкуш. Премьер-министрликке добуш берүү менен шайланса бир жөн эле», — деп жазды Сооронкулова Фейсбук барагында. Оппозиционерлердин билдирүүсү «Бириккен демократиялык кыймыл» 13-апрелде «Кумтөр: коррупциялык жана экологиялык тобокелчиликтер» деген темада өткөн жыйында Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген. Саясатчылар Кумтөрдү улутташтыруунун зарылдыгы тууралуу кайрылуу даярдашканын билдиришкен. Алардын пикиринде, канадалык компания менен келишим түзүү — Кыргызстан үчүн утулуш болот, анткени өлкө «Центеррадагы» акцияларынан баш тартып, экологиялык

**Generated:** илгелер» партиясынын басма сөз кызматынан билдиришти.<|end_of_text|>

Metadata: length=32, termination=eos, repetition=0.156250, invalid_decode=False

#### `heldout_kyrgyz_02` (heldout_kyrgyz)

**Prompt:**  чыгымдарды өз мойнуна алып жатат. «Бириккен демократиялык кыймыл» — бул кыргыз оппозициясынын жаңы түзү

**Reference:** мү, кыймылдын түзүлгөнү 31-мартта белгилүү болгон. Анын курамында Равшан Жээнбеков, Клара Сооронкулова, Бектур Асанов, Мамбетжунус Абылов, Кеңешбек Дүйшөбаев өңдүү саясатчылар, ошондой эле укук коргоочулар жана активисттер бар. Кыймылдын координатору — Мамбетжунус Абылов.<|end_of_text|>Макензи, Маккензи – Канаданын түндүк-батышындагы дарыя. Чоң Эрксиз көлүнөн башталып, Макензи ойдуңу аркылуу агат да Түндүк Муз океандын Бофорт деңизине куят. Өзүнүн Узундугу 1770 кмдей, Пис-Ривер менен (Финли дарыясынын башынан) 4250 км, алабынын аянты (Чоң Эрксиз көлүнүн алабына кирген Эрксиз, Пис-Ривер жана Атабаска дарыясынын системасы менен) 1804 миң км2. Ири куймалары: Лиард, Арктик-Ред-Ривер, Пил (сол), Чоң Аюулуу (оң). Жээги өтө саздак, карагай токою менен

**Generated:** лүшү. Бул өкүлдөрүнүн өкүлдөрүнүн өкүлдөрүнүн өк

Metadata: length=64, termination=max_new_tokens, repetition=0.781250, invalid_decode=False

#### `english_ood_00` (english_ood)

**Prompt:** Barovier&Toso is a Venetian company with a vocation and culture that is international, creating solutions of decorative lighting in Murano glass, all characterised by a personal and unique style. ​ Lighting solutions destined not only to furnish ambiances, but also to create emotions, transmit positivity, making each room

**Reference:**  unique. ​ Traditions and the future Murano glass crafting tradition, of which we have always been absolute protagonists, is for us the starting point from which to propose ideas, solutions, innovative and original projects that have allowed us to hand down values through the century old history of the founders’ family.<|end_of_text|>Beckton locksmith services are offered by us, locks in the City. A true 24 hour emergency service, we have the solution anytime Day or Night. When you call our Beckton Locksmith service we speak with an actual locksmith. Daniel or myself Paul. With a combined 25 years experience and knowledge in all kinds of locks and hardware. We will be more then happy to attend your property for a free survey Beckton Locksmith services are used by domestic and commercial customers. We have customers within the retail parks that we carry out many typical locksmithing duties. We also have a retail shop for customers to come to us if they wish to. Where you can purchase anything from novelty keys through to Locks and associated hardware. If you need us to attend to fit the item we would be more then willing as we are the local Beckton Locksmith. Call Locks in the City today on 07921 000672 for a fast and friendly response. Our locksmiths are fully insured and are members of the Master Locksmiths Association (MLA). The team has been inspected by the MLA. We are fully qualified & fully insured. Consequently, you can have complete peace of mind when your Locks in the City representative arrives at your address. You know that they will be qualified to do the work. We use non destructive entry methods where ever possible in emergency lock out situations.<|end_of_text|>match preview: The Exiles head to rugby lane and sutton & epsom, for the first time - in round 6 - London Welsh Rugby Club Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh Home Teams Men’s Men’s Rugby Performance Squad Women & Girls Women

**Generated:**  a unique space. The company is a pioneer in the field of decorative lighting in Murano glass, and has been able to create a unique style that is both contemporary and traditional. The company is a member of the Venetian Glass Association and the Italian Glass Association. The company is also a member of the European Union

Metadata: length=64, termination=max_new_tokens, repetition=0.390625, invalid_decode=False

#### `english_ood_01` (english_ood)

**Prompt:** ’s Rugby Women’s Squad Girls Youth & Minis Youth Rugby Mini Rugby Fixtures Latest News News & Media News Videos Gallery Podcast Membership Supporters club About Us Join Us Travel And Events The 400 Club Careers Hub Hub News The Hub The Idea Success Stories The Mentoring Experience Business Partners Contact The Hub Events & Hospitality LWRFC

**Reference:**  Choir Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh match preview: The Exiles head to rugby lane and sutton & epsom, for the first time – in round 6 News October 9, 2022 Following the disappointment of Round 5 and yet another top 4 side that Welsh will feel they should have beaten, Round 6 is yet another first time fixture for the London Welsh faithful. The Exiles fourth away trip of the season takes them to Ewell in Sutton and a side that Welsh meet for the first time in their history (something to which we have become accustom to since project reset). Sutton & Epsom has a similar longevity to that of London Welsh, playing their first matches in the 1883/84 season – the season prior to the founding of London Welsh. In recent seasons Sutton have found the going tough after promotion in 18/19 season from Level 5 to level 4 and National 2 South. Sutton are one of the sides to count themselves unlucky from the RFU calculations resulting from the 80% played season of covid riven 19/20. Sutton finished the played season 3rd from bottom and were relegated with no opportunity to complete the season and try to reel in 4th from bottom Westcliff. That said, Sutton were 14 points adrift of safety at that stage. In the first full season post Covid, Sutton finished 11th in South Premier last season (4th from bottom) and comfortably clear of Brighton in 12th. Although early in the season both sides will see this Round 6 fixture as crucial: Sutton currently sit just one place above Welsh in the league (9th) and with an identical tally of won 1 lost 4 (Sutton with 2 more bonus points to their name). Welsh will feel that they could and perhaps should have won all 4 of their previous fixtures, having taken 3 of the current top

**Generated:**  Archives 2022 2021 2020 2019 2018 2017 2016 2015 2014 2013 2012 2011 2010 2009 2008 2007 2006 2005 2004 2003 2002

Metadata: length=64, termination=max_new_tokens, repetition=0.765625, invalid_decode=False

#### `english_ood_02` (english_ood)

**Prompt:** 4 sides all the way. That all said, Welsh need league points in order to get their season properly underway. It is well accepted that Welsh have been victims of an almost unprecedented injury crisis in the early rounds, an issue that is only now seeing signs of abating. In the latest injury and availability news: Top

**Reference:**  try scorer Sion Cowdy remains on the side-lines; he is on track to return in Round 7 and is with the Ireland 7s squad in Spain this weekend. There will be a fitness test this week for Loosehead Griff Whitson after he too missed Round 6 following injury in Round 5. Will Ponty at second row suffered a concussion in Round 5 and will be undergo HIA protocol assessment this week as he clears the mandatory 3 weeks on the side-lines. Former Premiership star Ben West is almost ready for his first game of the season but Round 6 will come just that bit too early. In positive news: Matt Pollen returns at centre for the first time since Round 2, Wales u18 man Rhys Howells also returns in the centre for potentially his 53rd cap and following his wedding last weekend. Hywel Williams also returns at to contest the starting spot at 9 having been unavailable for round 5. Tom Williams played 80 minutes for the VETS (as their u35 player) against Wasps on Friday last as he continues his 6 month return to fitness, he may be in contention for a first game of the season. James Smith (broken shoulder) and Jack Anderson (dislocated knee) remain the long term absentees. Welsh could jump as high as 7th win a win in Round 6 and will want to continue their excellent form and growth that we have seen since round 4. Director of Men’s Rugby Cai Griffiths said of the game ” We have another big week in training this week with the squad returning to some form of normality and continuity, given the injuries that we have had. We expect this Round 6 match to be another tough contest with Sutton & Epsom coming close in many of their games so far this season. We have certainly felt like we are not far off where we need to be in the past few weeks and with a more settled team the squad will be keen to prove some points’. No doubt eyes will be on some of the big performers from the last few rounds: Samoan Kiwi Tololima Saavainaea has be excellent in his first three games and grabbed his

**Generated:**  scorer and captain, Rhys Evans, is out for the remainder of the season. Welsh have also lost to injury, the likes of Rhys Bevan, Rhys Griffiths, Rhys Jones, Rhys Powell, Rhys Thomas, Rhys Williams, and Rhys Williams. The Welsh squad is now down

Metadata: length=64, termination=max_new_tokens, repetition=0.468750, invalid_decode=False

### Step 2000

#### `heldout_kyrgyz_00` (heldout_kyrgyz)

**Prompt:** Министрлер кабинетинин төрагасынын орун басары Эдил Байсалов Фейсбук барагында «Бириккен демократиялык кыймылынын» билдир

**Reference:** үүсүнө жооп берди. Кыймыл 13-апрелде Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген болчу. «Оппозиция деп бул досторубузду атоо туура эмес. Равшан Бабыр уулу 2,5 миң добуш алган президенттик шайлоодо. Клара айымдыкы дурусурак: 1% — 14 миң добуш. Анан кайдагы оппозиция боло алышмак эле? Активист дегилечи сураныч. Ошол туура болот!» — деп жазды Байсалов. Байсалов кыймылдын мүчөлөрүн оппозиция деп эсептебесин билдирди. «Уят эле бирок. Саясатчы менен артист сезиши керек сахнадан качан түшүштү. Эптеп эле чырлап, кыпчылып, өрдөк жокто чулдук бий болуу аракети. Оппозиция болот, азыр деле бар. Бирок бул активисттер эмес. Бузуку деп айтпайын. Бирок Кумтөрдүн 100% элибизге кайтарылышына БУУ баш болуп, дүйнөлүк коомчулук суктанып, алкыш айтып атышса, куттуктап, бу

**Generated:** үүлөрүн ачыктоого жана анын көрсөтүүгө көзөмөл кылуу боюнча кызматташууга к

Metadata: length=64, termination=max_new_tokens, repetition=0.500000, invalid_decode=False

#### `heldout_kyrgyz_01` (heldout_kyrgyz)

**Prompt:** лар тескери каршы чыгып аткандары бузукулук эмей эмне», — дейт чиновник. Байсаловдун билдирүүсүнө «Бириккен дем

**Reference:** ократиялык кыймылдын» мүчөсү, саясатчы Клара Сооронкулова дагы жооп кайтарды. «Байсаловду сөгө албаганыма эч качан өкүнгөн эмесмин. Кайда, кайсы кеменин артынан “жибереримди” билбей турам. Байкуш. Премьер-министрликке добуш берүү менен шайланса бир жөн эле», — деп жазды Сооронкулова Фейсбук барагында. Оппозиционерлердин билдирүүсү «Бириккен демократиялык кыймыл» 13-апрелде «Кумтөр: коррупциялык жана экологиялык тобокелчиликтер» деген темада өткөн жыйында Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген. Саясатчылар Кумтөрдү улутташтыруунун зарылдыгы тууралуу кайрылуу даярдашканын билдиришкен. Алардын пикиринде, канадалык компания менен келишим түзүү — Кыргызстан үчүн утулуш болот, анткени өлкө «Центеррадагы» акцияларынан баш тартып, экологиялык

**Generated:** ократия кызматы» кыргызстандык эл аралык кызматташтыктын жетекчиси Аймактагы кызматынан кийин кыргызстандык эл

Metadata: length=64, termination=max_new_tokens, repetition=0.468750, invalid_decode=False

#### `heldout_kyrgyz_02` (heldout_kyrgyz)

**Prompt:**  чыгымдарды өз мойнуна алып жатат. «Бириккен демократиялык кыймыл» — бул кыргыз оппозициясынын жаңы түзү

**Reference:** мү, кыймылдын түзүлгөнү 31-мартта белгилүү болгон. Анын курамында Равшан Жээнбеков, Клара Сооронкулова, Бектур Асанов, Мамбетжунус Абылов, Кеңешбек Дүйшөбаев өңдүү саясатчылар, ошондой эле укук коргоочулар жана активисттер бар. Кыймылдын координатору — Мамбетжунус Абылов.<|end_of_text|>Макензи, Маккензи – Канаданын түндүк-батышындагы дарыя. Чоң Эрксиз көлүнөн башталып, Макензи ойдуңу аркылуу агат да Түндүк Муз океандын Бофорт деңизине куят. Өзүнүн Узундугу 1770 кмдей, Пис-Ривер менен (Финли дарыясынын башынан) 4250 км, алабынын аянты (Чоң Эрксиз көлүнүн алабына кирген Эрксиз, Пис-Ривер жана Атабаска дарыясынын системасы менен) 1804 миң км2. Ири куймалары: Лиард, Арктик-Ред-Ривер, Пил (сол), Чоң Аюулуу (оң). Жээги өтө саздак, карагай токою менен

**Generated:** лгөн кыймыл. Бул кыймылдын негизинде кыргызстандыктардын өз ара кырдаалдын өзүнчөлүгүн�

Metadata: length=64, termination=max_new_tokens, repetition=0.515625, invalid_decode=True

#### `english_ood_00` (english_ood)

**Prompt:** Barovier&Toso is a Venetian company with a vocation and culture that is international, creating solutions of decorative lighting in Murano glass, all characterised by a personal and unique style. ​ Lighting solutions destined not only to furnish ambiances, but also to create emotions, transmit positivity, making each room

**Reference:**  unique. ​ Traditions and the future Murano glass crafting tradition, of which we have always been absolute protagonists, is for us the starting point from which to propose ideas, solutions, innovative and original projects that have allowed us to hand down values through the century old history of the founders’ family.<|end_of_text|>Beckton locksmith services are offered by us, locks in the City. A true 24 hour emergency service, we have the solution anytime Day or Night. When you call our Beckton Locksmith service we speak with an actual locksmith. Daniel or myself Paul. With a combined 25 years experience and knowledge in all kinds of locks and hardware. We will be more then happy to attend your property for a free survey Beckton Locksmith services are used by domestic and commercial customers. We have customers within the retail parks that we carry out many typical locksmithing duties. We also have a retail shop for customers to come to us if they wish to. Where you can purchase anything from novelty keys through to Locks and associated hardware. If you need us to attend to fit the item we would be more then willing as we are the local Beckton Locksmith. Call Locks in the City today on 07921 000672 for a fast and friendly response. Our locksmiths are fully insured and are members of the Master Locksmiths Association (MLA). The team has been inspected by the MLA. We are fully qualified & fully insured. Consequently, you can have complete peace of mind when your Locks in the City representative arrives at your address. You know that they will be qualified to do the work. We use non destructive entry methods where ever possible in emergency lock out situations.<|end_of_text|>match preview: The Exiles head to rugby lane and sutton & epsom, for the first time - in round 6 - London Welsh Rugby Club Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh Home Teams Men’s Men’s Rugby Performance Squad Women & Girls Women

**Generated:**  a unique space. The company is based on the concept of "light as art", where the glass is the protagonist, the light is the medium, and the designer is the artist. The company is committed to the creation of unique and exclusive products, which are the result of a long and careful work of research and experimentation

Metadata: length=64, termination=max_new_tokens, repetition=0.390625, invalid_decode=False

#### `english_ood_01` (english_ood)

**Prompt:** ’s Rugby Women’s Squad Girls Youth & Minis Youth Rugby Mini Rugby Fixtures Latest News News & Media News Videos Gallery Podcast Membership Supporters club About Us Join Us Travel And Events The 400 Club Careers Hub Hub News The Hub The Idea Success Stories The Mentoring Experience Business Partners Contact The Hub Events & Hospitality LWRFC

**Reference:**  Choir Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh match preview: The Exiles head to rugby lane and sutton & epsom, for the first time – in round 6 News October 9, 2022 Following the disappointment of Round 5 and yet another top 4 side that Welsh will feel they should have beaten, Round 6 is yet another first time fixture for the London Welsh faithful. The Exiles fourth away trip of the season takes them to Ewell in Sutton and a side that Welsh meet for the first time in their history (something to which we have become accustom to since project reset). Sutton & Epsom has a similar longevity to that of London Welsh, playing their first matches in the 1883/84 season – the season prior to the founding of London Welsh. In recent seasons Sutton have found the going tough after promotion in 18/19 season from Level 5 to level 4 and National 2 South. Sutton are one of the sides to count themselves unlucky from the RFU calculations resulting from the 80% played season of covid riven 19/20. Sutton finished the played season 3rd from bottom and were relegated with no opportunity to complete the season and try to reel in 4th from bottom Westcliff. That said, Sutton were 14 points adrift of safety at that stage. In the first full season post Covid, Sutton finished 11th in South Premier last season (4th from bottom) and comfortably clear of Brighton in 12th. Although early in the season both sides will see this Round 6 fixture as crucial: Sutton currently sit just one place above Welsh in the league (9th) and with an identical tally of won 1 lost 4 (Sutton with 2 more bonus points to their name). Welsh will feel that they could and perhaps should have won all 4 of their previous fixtures, having taken 3 of the current top

**Generated:**  Calendar 2024 LWRFC Calendar 2023 LWRFC Calendar 2022 LWRFC Calendar 2021 LWRFC Calendar 2020 LWRFC Calendar 2019 LWRFC Calendar 2018 LWRFC Calendar 2017 LWRFC Calendar 2016 LWRFC Calendar

Metadata: length=64, termination=max_new_tokens, repetition=0.750000, invalid_decode=False

#### `english_ood_02` (english_ood)

**Prompt:** 4 sides all the way. That all said, Welsh need league points in order to get their season properly underway. It is well accepted that Welsh have been victims of an almost unprecedented injury crisis in the early rounds, an issue that is only now seeing signs of abating. In the latest injury and availability news: Top

**Reference:**  try scorer Sion Cowdy remains on the side-lines; he is on track to return in Round 7 and is with the Ireland 7s squad in Spain this weekend. There will be a fitness test this week for Loosehead Griff Whitson after he too missed Round 6 following injury in Round 5. Will Ponty at second row suffered a concussion in Round 5 and will be undergo HIA protocol assessment this week as he clears the mandatory 3 weeks on the side-lines. Former Premiership star Ben West is almost ready for his first game of the season but Round 6 will come just that bit too early. In positive news: Matt Pollen returns at centre for the first time since Round 2, Wales u18 man Rhys Howells also returns in the centre for potentially his 53rd cap and following his wedding last weekend. Hywel Williams also returns at to contest the starting spot at 9 having been unavailable for round 5. Tom Williams played 80 minutes for the VETS (as their u35 player) against Wasps on Friday last as he continues his 6 month return to fitness, he may be in contention for a first game of the season. James Smith (broken shoulder) and Jack Anderson (dislocated knee) remain the long term absentees. Welsh could jump as high as 7th win a win in Round 6 and will want to continue their excellent form and growth that we have seen since round 4. Director of Men’s Rugby Cai Griffiths said of the game ” We have another big week in training this week with the squad returning to some form of normality and continuity, given the injuries that we have had. We expect this Round 6 match to be another tough contest with Sutton & Epsom coming close in many of their games so far this season. We have certainly felt like we are not far off where we need to be in the past few weeks and with a more settled team the squad will be keen to prove some points’. No doubt eyes will be on some of the big performers from the last few rounds: Samoan Kiwi Tololima Saavainaea has be excellent in his first three games and grabbed his

**Generated:**  scorer Rhys Bennett is out for the remainder of the season after suffering a knee injury. Welsh have been forced to play without their captain, 30-year-old Chris Davies, for the last two games. Davies is out for the season after suffering a knee injury. Welsh have been forced to play without their captain,

Metadata: length=64, termination=max_new_tokens, repetition=0.437500, invalid_decode=False

### Step 5000

#### `heldout_kyrgyz_00` (heldout_kyrgyz)

**Prompt:** Министрлер кабинетинин төрагасынын орун басары Эдил Байсалов Фейсбук барагында «Бириккен демократиялык кыймылынын» билдир

**Reference:** үүсүнө жооп берди. Кыймыл 13-апрелде Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген болчу. «Оппозиция деп бул досторубузду атоо туура эмес. Равшан Бабыр уулу 2,5 миң добуш алган президенттик шайлоодо. Клара айымдыкы дурусурак: 1% — 14 миң добуш. Анан кайдагы оппозиция боло алышмак эле? Активист дегилечи сураныч. Ошол туура болот!» — деп жазды Байсалов. Байсалов кыймылдын мүчөлөрүн оппозиция деп эсептебесин билдирди. «Уят эле бирок. Саясатчы менен артист сезиши керек сахнадан качан түшүштү. Эптеп эле чырлап, кыпчылып, өрдөк жокто чулдук бий болуу аракети. Оппозиция болот, азыр деле бар. Бирок бул активисттер эмес. Бузуку деп айтпайын. Бирок Кумтөрдүн 100% элибизге кайтарылышына БУУ баш болуп, дүйнөлүк коомчулук суктанып, алкыш айтып атышса, куттуктап, бу

**Generated:** үүсүнөн кийин өзүнүн өзүнүн жакында өзүнүн жакында өзүнүн

Metadata: length=64, termination=max_new_tokens, repetition=0.765625, invalid_decode=False

#### `heldout_kyrgyz_01` (heldout_kyrgyz)

**Prompt:** лар тескери каршы чыгып аткандары бузукулук эмей эмне», — дейт чиновник. Байсаловдун билдирүүсүнө «Бириккен дем

**Reference:** ократиялык кыймылдын» мүчөсү, саясатчы Клара Сооронкулова дагы жооп кайтарды. «Байсаловду сөгө албаганыма эч качан өкүнгөн эмесмин. Кайда, кайсы кеменин артынан “жибереримди” билбей турам. Байкуш. Премьер-министрликке добуш берүү менен шайланса бир жөн эле», — деп жазды Сооронкулова Фейсбук барагында. Оппозиционерлердин билдирүүсү «Бириккен демократиялык кыймыл» 13-апрелде «Кумтөр: коррупциялык жана экологиялык тобокелчиликтер» деген темада өткөн жыйында Кыргызстан Кумтөр боюнча Centerra Gold Inc. менен тынчтык келишимин түзүүдөн баш тартышы керек деп билдирген. Саясатчылар Кумтөрдү улутташтыруунун зарылдыгы тууралуу кайрылуу даярдашканын билдиришкен. Алардын пикиринде, канадалык компания менен келишим түзүү — Кыргызстан үчүн утулуш болот, анткени өлкө «Центеррадагы» акцияларынан баш тартып, экологиялык

**Generated:** ократиялык партиясы» партиясынын төрагасы болуп жатканы аныкталган. Бирок, партиянын төрагасынын орун басары болуп жатканы аныктал

Metadata: length=64, termination=max_new_tokens, repetition=0.468750, invalid_decode=False

#### `heldout_kyrgyz_02` (heldout_kyrgyz)

**Prompt:**  чыгымдарды өз мойнуна алып жатат. «Бириккен демократиялык кыймыл» — бул кыргыз оппозициясынын жаңы түзү

**Reference:** мү, кыймылдын түзүлгөнү 31-мартта белгилүү болгон. Анын курамында Равшан Жээнбеков, Клара Сооронкулова, Бектур Асанов, Мамбетжунус Абылов, Кеңешбек Дүйшөбаев өңдүү саясатчылар, ошондой эле укук коргоочулар жана активисттер бар. Кыймылдын координатору — Мамбетжунус Абылов.<|end_of_text|>Макензи, Маккензи – Канаданын түндүк-батышындагы дарыя. Чоң Эрксиз көлүнөн башталып, Макензи ойдуңу аркылуу агат да Түндүк Муз океандын Бофорт деңизине куят. Өзүнүн Узундугу 1770 кмдей, Пис-Ривер менен (Финли дарыясынын башынан) 4250 км, алабынын аянты (Чоң Эрксиз көлүнүн алабына кирген Эрксиз, Пис-Ривер жана Атабаска дарыясынын системасы менен) 1804 миң км2. Ири куймалары: Лиард, Арктик-Ред-Ривер, Пил (сол), Чоң Аюулуу (оң). Жээги өтө саздак, карагай токою менен

**Generated:** лүштөрүнүн бири. Бул кыймыл 2010-жылдан бери «Кыргызстан» партиясынын лидери Алмазбек Атамбаевдин уюштуруу кат

Metadata: length=64, termination=max_new_tokens, repetition=0.140625, invalid_decode=False

#### `english_ood_00` (english_ood)

**Prompt:** Barovier&Toso is a Venetian company with a vocation and culture that is international, creating solutions of decorative lighting in Murano glass, all characterised by a personal and unique style. ​ Lighting solutions destined not only to furnish ambiances, but also to create emotions, transmit positivity, making each room

**Reference:**  unique. ​ Traditions and the future Murano glass crafting tradition, of which we have always been absolute protagonists, is for us the starting point from which to propose ideas, solutions, innovative and original projects that have allowed us to hand down values through the century old history of the founders’ family.<|end_of_text|>Beckton locksmith services are offered by us, locks in the City. A true 24 hour emergency service, we have the solution anytime Day or Night. When you call our Beckton Locksmith service we speak with an actual locksmith. Daniel or myself Paul. With a combined 25 years experience and knowledge in all kinds of locks and hardware. We will be more then happy to attend your property for a free survey Beckton Locksmith services are used by domestic and commercial customers. We have customers within the retail parks that we carry out many typical locksmithing duties. We also have a retail shop for customers to come to us if they wish to. Where you can purchase anything from novelty keys through to Locks and associated hardware. If you need us to attend to fit the item we would be more then willing as we are the local Beckton Locksmith. Call Locks in the City today on 07921 000672 for a fast and friendly response. Our locksmiths are fully insured and are members of the Master Locksmiths Association (MLA). The team has been inspected by the MLA. We are fully qualified & fully insured. Consequently, you can have complete peace of mind when your Locks in the City representative arrives at your address. You know that they will be qualified to do the work. We use non destructive entry methods where ever possible in emergency lock out situations.<|end_of_text|>match preview: The Exiles head to rugby lane and sutton & epsom, for the first time - in round 6 - London Welsh Rugby Club Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh Home Teams Men’s Men’s Rugby Performance Squad Women & Girls Women

**Generated:**  a unique space. The company is based on the concept of "light as art", creating pieces of art that are both functional and decorative, with a strong focus on the quality of the product, the design and the attention to detail. The company is a family business, with a team of artisans and designers who share the

Metadata: length=64, termination=max_new_tokens, repetition=0.343750, invalid_decode=False

#### `english_ood_01` (english_ood)

**Prompt:** ’s Rugby Women’s Squad Girls Youth & Minis Youth Rugby Mini Rugby Fixtures Latest News News & Media News Videos Gallery Podcast Membership Supporters club About Us Join Us Travel And Events The 400 Club Careers Hub Hub News The Hub The Idea Success Stories The Mentoring Experience Business Partners Contact The Hub Events & Hospitality LWRFC

**Reference:**  Choir Sponsors & Partners Contacts & Policies Club Contacts Club Policies & Resources Incident Reporting Form First Aid Guidance Organising Team Rugby Tours Club Safeguarding Guidance Codes of Conduct Equality, Diversity and Inclusiveness Policy Data Privacy Notice Photo & Video Policy Club Rules RFU Discipline Regulations Communications & Media Policy Club Shop About London Welsh match preview: The Exiles head to rugby lane and sutton & epsom, for the first time – in round 6 News October 9, 2022 Following the disappointment of Round 5 and yet another top 4 side that Welsh will feel they should have beaten, Round 6 is yet another first time fixture for the London Welsh faithful. The Exiles fourth away trip of the season takes them to Ewell in Sutton and a side that Welsh meet for the first time in their history (something to which we have become accustom to since project reset). Sutton & Epsom has a similar longevity to that of London Welsh, playing their first matches in the 1883/84 season – the season prior to the founding of London Welsh. In recent seasons Sutton have found the going tough after promotion in 18/19 season from Level 5 to level 4 and National 2 South. Sutton are one of the sides to count themselves unlucky from the RFU calculations resulting from the 80% played season of covid riven 19/20. Sutton finished the played season 3rd from bottom and were relegated with no opportunity to complete the season and try to reel in 4th from bottom Westcliff. That said, Sutton were 14 points adrift of safety at that stage. In the first full season post Covid, Sutton finished 11th in South Premier last season (4th from bottom) and comfortably clear of Brighton in 12th. Although early in the season both sides will see this Round 6 fixture as crucial: Sutton currently sit just one place above Welsh in the league (9th) and with an identical tally of won 1 lost 4 (Sutton with 2 more bonus points to their name). Welsh will feel that they could and perhaps should have won all 4 of their previous fixtures, having taken 3 of the current top

**Generated:**  Partners LWRFC Products & Services LWRFC Membership LWRFC News LWRFC Photos LWRFC Videos LWRFC Webinars LWRFC Youth LWRFC Coaching Courses LWRFC Coaching Courses LWRFC Coaching Courses LWRFC Coaching Courses LWRFC Coaching Courses LWRFC Coaching Courses LWR

Metadata: length=64, termination=max_new_tokens, repetition=0.750000, invalid_decode=False

#### `english_ood_02` (english_ood)

**Prompt:** 4 sides all the way. That all said, Welsh need league points in order to get their season properly underway. It is well accepted that Welsh have been victims of an almost unprecedented injury crisis in the early rounds, an issue that is only now seeing signs of abating. In the latest injury and availability news: Top

**Reference:**  try scorer Sion Cowdy remains on the side-lines; he is on track to return in Round 7 and is with the Ireland 7s squad in Spain this weekend. There will be a fitness test this week for Loosehead Griff Whitson after he too missed Round 6 following injury in Round 5. Will Ponty at second row suffered a concussion in Round 5 and will be undergo HIA protocol assessment this week as he clears the mandatory 3 weeks on the side-lines. Former Premiership star Ben West is almost ready for his first game of the season but Round 6 will come just that bit too early. In positive news: Matt Pollen returns at centre for the first time since Round 2, Wales u18 man Rhys Howells also returns in the centre for potentially his 53rd cap and following his wedding last weekend. Hywel Williams also returns at to contest the starting spot at 9 having been unavailable for round 5. Tom Williams played 80 minutes for the VETS (as their u35 player) against Wasps on Friday last as he continues his 6 month return to fitness, he may be in contention for a first game of the season. James Smith (broken shoulder) and Jack Anderson (dislocated knee) remain the long term absentees. Welsh could jump as high as 7th win a win in Round 6 and will want to continue their excellent form and growth that we have seen since round 4. Director of Men’s Rugby Cai Griffiths said of the game ” We have another big week in training this week with the squad returning to some form of normality and continuity, given the injuries that we have had. We expect this Round 6 match to be another tough contest with Sutton & Epsom coming close in many of their games so far this season. We have certainly felt like we are not far off where we need to be in the past few weeks and with a more settled team the squad will be keen to prove some points’. No doubt eyes will be on some of the big performers from the last few rounds: Samoan Kiwi Tololima Saavainaea has be excellent in his first three games and grabbed his

**Generated:**  of the table: 1. Cardiff City 2. Swansea City 3. Newport County 4. Bristol City 5. Brentford 6. Brighton & Hove 7. Leeds United 8. Bristol Rovers 9. Milton Keynes Dons 10. Stockport County 11. W

Metadata: length=64, termination=max_new_tokens, repetition=0.375000, invalid_decode=False

## Interpretation

Completed after inspection of all checkpoint outputs.

- Step 200 reproduces the earlier failure: Kyrgyz continuations loop, while English is partly topical but includes copied season/list text.
- Step 1000 improves heldout PPL to `8.12`; one Kyrgyz sample becomes locally plausible, but another remains a loop and English still contains archive/list copying.
- Step 2000 improves heldout PPL to `6.90`; some heldout Kyrgyz text is locally grammatical, but FLORES remains inconsistent and English still repeats web-calendar content.
- Step 5000 reaches `2,555,000` supervised tokens and heldout PPL `6.11`, but the three Kyrgyz outputs are not consistently coherent: one is visibly repetitive, one is copied web text, and one is only locally plausible. English remains partly topical but contains repeated navigation/list continuations. No checkpoint demonstrates robust coherent Kyrgyz continuation.
- English OOD PPL worsens from `13.81` at step 200 to `16.94` at step 5000, consistent with increasing specialization without a clean capability gain.
- **LoRA gate: not passed.** LoRA remains insufficiently coherent through 5000 steps despite lower Kyrgyz PPL. Phase 4 matched Axis training was therefore not run; the next leading issue is shared corpus/packing/tokenizer/objective/generalization quality, not Axis ownership.
- This LoRA curve is diagnostic control only and does not propose incorporating LoRA into Grafting.

The first exposure launch stopped at step 200 while writing progress metrics because it referenced future checkpoint losses (`IndexError: list index out of range`). The run was restarted from step 1 with the corrected progress writer and completed all four checkpoints. The successful command was:

```text
ssh -p 31101 root@36.150.116.206 "timeout 8400 /opt/venv/bin/python /workspace/GLT/experiments/kyrgyz_lora_exposure/run_exposure.py"
```
