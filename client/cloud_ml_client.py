import numpy as np
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

from data_utils import constants
from data_utils.get_chunks import get_chunks
from data_utils.transform_data import pad_sequences, transform_data


class Client(object):
    def __init__(self, word_tokenizer, char_tokenizer, credential_json, model_name, project_name, version):
        credentials = GoogleCredentials.get_application_default()
        self.service = discovery.build('ml', 'v1', credentials=credentials)
        self.name = 'projects/{}/models/{}'.format(project_name, model_name)
        if version is not None:
            self.name += '/versions/{}'.format(version)
        self.word_tokenizer = word_tokenizer
        self.char_tokenizer = char_tokenizer

    def send_requests(self, texts):
        transformed = [
            transform_data(text, self.word_tokenizer, self.char_tokenizer) for text in texts
        ]
        seq_len = [x[1] for x in transformed]
        words = [x[0] for x in transformed]
        chars = [x[2] for x in transformed]
        word_lengths = pad_sequences(
            [x[3] for x in transformed], max(seq_len))
        max_char_len = np.max(word_lengths)
        padded_chars = np.zeros(
            [len(texts), max(seq_len), max_char_len], dtype=np.int32)
        for p1, c1 in zip(padded_chars, chars):
            for i, c2 in enumerate(c1):
                p1[i][:len(c2)] = c2
        words = pad_sequences(words, max(seq_len))
        instances = [
            {
                'word_ids': x[0].tolist(),
                'char_ids': x[1].tolist(),
                'sequence_length': x[2],
                'word_length': x[3].tolist()
            } for x in zip(words, padded_chars, seq_len, word_lengths)
        ]
        response = self.service.projects().predict(
            name=self.name,
            body={'instances': instances}
        ).execute()
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response


if __name__ == '__main__':
    import pickle
    with open('client/word_tokenizer.pkl', 'rb') as file:
        word_tokenizer = pickle.load(file)
    with open('client/char_tokenizer.pkl', 'rb') as file:
        char_tokenizer = pickle.load(file)
    texts = ["""Cần thuê nhà để kinh doanh bún phở. Mặt đường hoặc ngõ to, có chỗ để xe tại quận Hai Bà Trưng, Thanh Xuân, Hoàng Mai hoặc Đống Đa. Diện tích >=25m2, giá thuê 8 - 12 triệu, thanh toán 3 tháng /lần.""",
             """Bán gấp trong năm nhà 2 MT đường Đoàn Thị Điểm, P1, Phú Nhuận
Vị trí: Cách MT Phan Đăng Lưu chỉ 40m, cách ngã tư Phú Nhuận 100m. Nằm khu vực trung tâm, xung quanh đầy đủ các tiện ích. Cơ sở hạ tầng đầy đủ.
Kết cấu: Nhà 1T, 1L cũ nhưng nội thất đẹp, góc 2 MT dễ kinh doanh buôn bán
DT: 4.25x13m, đất vuông vức, không lộ giới
Pháp lý: Sổ hồng chính chủ, đầy đủ pháp lý, sổ mới năm 2017
Giá bán: 12 tỷ, thương lượng chút xíu lấy lộc.
LH xem nhà chính chủ 0967463475 (Mr. Hóa)""", "- Cho thuê nhà nguyên căn hẻm xe hơi ni sư huỳnh liên, Phường 10, quận Tân Bình. 5x18m. 1 Trệt, 1 lầu, sân thượng rộng rãi, 2 phòng ngủ, 2 tolet . Trệt trống suốt, cầu thang 3/4 căn nhà.\n- Thích hợp  khai thác làm spa, kinh doanh online, ở hộ gia đình, cà phê, lớp dạy thêm,....\n- Gía thuê chỉ 19tr/th", "Do mình chuẩn bị chuyển vào ktx nên phòng mình đang ở còn trống 1 chổ.\nTiền phòng: 1tr850/tháng, ở được 3 người (hiện tại có 2 người, đã ra trường đi làm), có tủ lạnh.\nĐịa chỉ: 58/7A1. đường Đồng Nai, CX bắc hải. cách cổng 3 THT tầm 300m.\nGiờ giấc: 6h00-22h30, về trễ nhớ báo trước.\nĐiện nước theo giá nhà nước, Wifi (mới lắp mới, ping 8ms) 30k/tháng.\nBạn nào có nhu cầu thì inbox mình (0932069143)", "Địa chỉ: 88 Bạch Đằng, phường 2, quận Tân Bình\nQuy mô tòa nhà: 1 hầm, 1 trệt, 1 lửng, 5 tầng, sân thượng. Thang máy xuống tới tầng hầm.\nCác diện tích còn trống cần cho thuê:\n- 70m2 lầu 1 giá 19tr\n- 30m2 lầu 6 giá 8tr5 ( đã có sẵn kính ngăn, view ở giữa)\nGiá trên đã bao gồm phí dịch vụ, chưa bao gồm VAT. \nGiá thuê và thời gian setup có thể thương lượng và cân đối tùy theo nguyện vọng của khách hàng. \n* Phí giữ xe: 100 nghìn/chiếc/tháng. Xe hơi có thể đâu ở công viên sát bên tòa nhà. \n* Giá điện: 3500đ/kwh. \n* Đặt cọc 3 tháng, thanh toán từng tháng. \nMọi chi tiết xin liên hệ: Mr.Bảo 0898797789", "Nhà cho thuê nguyên căn mặt tiền đường Yên Thế, phường 2, quận Tân Bình.\nNhà rộng 4.4m dài 19.6m 1 trệt 1 lửng 3 lầu và 1 sân thượng.\ncó 6 phòng, 7 tolet, ban công riêng.\nTrệt và lửng trống suốt cầu thang 3/4 căn nhà.\nThích hợp làm văn phòng, kinh doanh, logistic,..\nGía chỉ 28 triệu/tháng"]
    client = Client(word_tokenizer, char_tokenizer,
                    'client/credential.json', 'ree', 'real-estate-extraction-196916', 'v11')
    print(client.send_requests(texts))
