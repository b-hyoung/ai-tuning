
import json
import os

def refine_training_data():
    original_path = './data/robot_agent_train.jsonl'
    refined_path = './data/robot_agent_train_refined.jsonl'
    
    refined_samples = []

    with open(original_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # 'output'은 문자열이므로 다시 JSON으로 파싱
            output_data = json.loads(data['output'])
            
            gui_message = output_data.get('gui_message')
            
            if gui_message:
                # voice_instruction을 gui_message 기반으로 새로 생성
                # 간단한 규칙: 명사형 메시지를 완전한 문장으로 변경
                if not gui_message.endswith(('요', '니다', '습니다', '시오', '세요')):
                    # 예: "탐색 시작" -> "탐색 시작합니다."
                    # 예: "생존자 발견" -> "생존자 발견했습니다."
                    # 상황에 따라 더 정교한 규칙이 필요할 수 있음
                    if '발견' in gui_message:
                         voice_instruction = gui_message.replace('발견', ' 발견했습니다')
                    elif '시작' in gui_message:
                        voice_instruction = gui_message.replace('시작', ' 시작합니다')
                    elif '보고' in gui_message:
                        voice_instruction = gui_message.replace('보고', ' 보고합니다')
                    elif '필요' in gui_message:
                        voice_instruction = gui_message.replace('필요', ' 필요합니다')
                    else:
                        voice_instruction = gui_message + "입니다"
                else:
                    # 이미 문장 형태이면 그대로 사용
                    voice_instruction = gui_message

                # 'voice_instruction' 업데이트
                output_data['voice_instruction'] = voice_instruction
            
            # 수정된 output_data를 다시 문자열로 변환하여 'output' 필드에 할당
            data['output'] = json.dumps(output_data, ensure_ascii=False)
            refined_samples.append(data)

    # 새로운 파일에 저장
    with open(refined_path, 'w', encoding='utf-8') as f:
        for sample in refined_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"데이터 정제 완료! '{refined_path}'에 저장되었습니다.")
    print(f"총 {len(refined_samples)}개의 샘플이 처리되었습니다.")

if __name__ == '__main__':
    refine_training_data()
