<!--
SPDX-FileCopyrightText: Copyright 2026 OSSP Router contributors
SPDX-License-Identifier: Apache-2.0
-->

# 모델·산출물 출처

## n-gram 분류기

- 파일: `n-gram/ngram_classifier_v2_float32.joblib`
- SHA-256: `87ae23d6207443c422c11e1503ac0f56262f97a22be1fcf2586d95f23d006a4e`
- 용도: prompt에서 response language, request type, content domain, answer format 특징 추출
- 학습 자료: 대회가 제공한 reviewed Train 분류 라벨 1차 학습 + Dev 분류 라벨 2차 학습
- 외부 추론 모델: 없음
- 실행 라이선스: 프로젝트 Apache-2.0

## 라우터 회귀·검색 산출물

- 파일: `라우터 분류/artifacts/router_model.joblib`
- SHA-256: `bc4c44dd15edacf20415994f1881d3c85c6da643cba611698db80e4148406359`
- 학습 자료: 공식 Train 1,760문항 outcomes, reviewed 분류 라벨 + Dev 880문항 outcomes(2차 학습)
- 정책 선택: Train 적합 후 Dev 분리 검증
- 최종 품질 모델: 정책 확정 뒤 공개 Train+Dev 2,640문항으로 재적합
- 토큰 모델: Train 적합과 Dev 총합·길이-band 보정 상태를 유지
- 실행 라이선스: 프로젝트 Apache-2.0

## DeBERTa tokenizer

라우터 산출물에는 DeBERTa 신경망 가중치가 없고 tokenizer 규칙 JSON만 포함한다.

- upstream: `microsoft/deberta-v3-small`
- 공개 위치: https://huggingface.co/microsoft/deberta-v3-small
- 확인한 고정 upstream 스냅샷: `b25b093541eedd589b3fd60c30142da149189960`
- 로컬 변환 tokenizer JSON SHA-256: `e9ddd853a4a31023721c993458663f2019e3154a05dacf0d2b3c217e53d9bdc8`
- 변환 산출물: Hugging Face fast-tokenizer JSON. 독립 파일이 아니라
  `router_model.joblib`의 세 입력 토큰 예측기에 같은 JSON 문자열로 내장되어
  실행 중 외부 파일이나 다운로드를 요구하지 않는다.
- 라이선스: MIT
- 용도: 입력 토큰 수 규칙 특징. DeBERTa 본체 추론에는 사용하지 않음.

고정 스냅샷의 SentencePiece 어휘에서 변환된 JSON이며, 산출물
SHA-256과 upstream revision은 이 문서에 고정했다.

## 제출 포함 범위

저장소에는 평가 실행에 필요한 추론 코드와 학습 산출물만 포함한다. 공개
Train/Dev 원본, outcomes, 실험 결과, 중간 보고서와 학습용 스크립트는 제출
이미지와 참가자 추가 파일에서 제외한다.

## 컨테이너 기반 이미지

- image: `python:3.11.15-slim-bookworm`
- multi-platform digest: `sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3`
- linux/arm64 manifest digest: `sha256:ecb0ac954790dd64a0d518d699b9c61a91780c42b0d877c802dbaffd04db66f9`
- 선택 이유: scikit-learn 1.9.0은 Debian 계열 manylinux ARM64 wheel을 제공하지만 Alpine musllinux ARM64 wheel은 제공하지 않음
