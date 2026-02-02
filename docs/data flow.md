## 原始数据集

来自`test.json`

```json
{
    "id": "WebQTest-0",
    "question": "what does jamaican people speak",
    "entities": [
        4648
    ],
    "answers": [
        {
            "kb_id": "m.01428y",
            "text": "Jamaican English"
        },
        {
            "kb_id": "m.04ygk0",
            "text": "Jamaican Creole English Language"
        }
    ],
    "subgraph": {
        "tuples": [
            [
                4648,
                430,
                77418
            ],
            [
                4648,
                448,
                77419
            ], .....
        }
}
```


## XX处理后

来自`test.info`

```json
{
        "question": "lou seal is the mascot for the team that last won the world series when ? ",
        "0": {},
        "1": {},
        "answers": [
            "m.0117q3yz"
        ],
        "precison": 0.3333333333333333,
        "recall": 1.0,
        "f1": 0.5,
        "hit": 1.0,
        "em": 1,
        "cand": [
            [
                "m.0117q3yz",
                0.5999045372009277
            ],
            [
                "m.09gnk2r",
                0.200139582157135
            ],
            [
                "m.0ds8qct",
                0.19965460896492004
            ]
        ]
}
```