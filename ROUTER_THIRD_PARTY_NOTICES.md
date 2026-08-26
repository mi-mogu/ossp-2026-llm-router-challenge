<!--
SPDX-FileCopyrightText: Copyright 2026 OSSP Router contributors
SPDX-License-Identifier: Apache-2.0
-->

# Third-party notices

이 프로젝트의 자체 코드는 Apache-2.0으로 배포한다. 실행 산출물과 컨테이너는 다음 공개 구성요소를 사용한다.

| 구성요소 | 고정 버전 | 라이선스 | 용도 |
| --- | --- | --- | --- |
| Python | 3.11.15 | PSF-2.0 | 런타임 |
| Debian Bookworm slim | Python 이미지 고정 digest 참조 | Debian 구성요소별 라이선스 | 기반 이미지 |
| joblib | 1.5.3 | BSD-3-Clause | 산출물 직렬화 |
| narwhals | 2.0.1 | MIT | scikit-learn 런타임 의존성 |
| NumPy | 2.4.6 | BSD-3-Clause | 수치 배열 |
| SciPy | 1.17.1 | BSD-3-Clause | 희소 행렬·최적화 |
| scikit-learn | 1.9.0 | BSD-3-Clause | n-gram·회귀·분류 |
| threadpoolctl | 3.6.0 | BSD-3-Clause | 수치 스레드 제어 |
| Hugging Face tokenizers | 0.23.1 | Apache-2.0 | tokenizer 규칙 실행 |
| microsoft/deberta-v3-small tokenizer | 고정 스냅샷은 `MODEL_PROVENANCE.md` 참조 | MIT | 입력 토큰 특징 |

DeBERTa tokenizer 저작권은 Microsoft 및 해당 기여자에게 있다. 원본 MIT 라이선스와 모델 카드는 https://huggingface.co/microsoft/deberta-v3-small 에서 확인할 수 있다.

각 Python 패키지의 전체 라이선스 텍스트는 해당 배포물의 metadata와 upstream 저장소를 따른다. 최종 SBOM은 실제 `linux/arm64` 이미지 빌드 후 그 이미지에서 생성해야 한다.
