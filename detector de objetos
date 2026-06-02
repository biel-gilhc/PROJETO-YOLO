import cv2
import streamlit as st
from ultralytics import YOLO


def main():
    # 1. Configuração da página do Streamlit
    st.set_page_config(page_title="Scanner com Yolo", layout="centered")
    st.title("Scanner com YOLO")
    st.write(
        "Protótipo funcional para detecção de objetos em tempo real utilizando a câmera."
    )

    # 2. Carregamento do modelo YOLO (utilizando cache para performance)
    @st.cache_resource
    def load_model():
        # Carrega o modelo YOLOv8 inicial (nano), ideal para deploys leves
        return YOLO("yolov8n.pt")

    model = load_model()

    # 4. Utiliza o botão para abrir/ativar a câmera
    # O componente 'camera_input' do Streamlit abre a câmera nativamente
    img_file_buffer = st.camera_input("Tirar uma foto para escanear")

    # 3. Processamento e detecção dos objetos se a imagem for capturada
    if img_file_buffer is not None:
        # Lê a imagem capturada pelo Streamlit
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(
            import_image_bytes(bytes_data), cv2.IMREAD_COLOR
        )

        # Realiza a predição com o modelo YOLO carregado
        results = model(cv2_img)

        # Renderiza os resultados com os bounding boxes desenhados na imagem
        annotated_frame = results[0].plot()

        # Converte de BGR (OpenCV) para RGB (Streamlit) para exibição correta das cores
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Exibe o resultado final na interface do usuário
        st.subheader("Resultado da Detecção:")
        st.image(
            annotated_frame_rgb,
            caption="Objetos Detectados",
            use_container_width=True,
        )


def import_image_bytes(bytes_data):
    # Função auxiliar para converter os bytes da imagem em formato legível pelo OpenCV
    import numpy as np

    return np.frombuffer(bytes_data, np.uint8)


if __name__ == "__main__":
    main()
