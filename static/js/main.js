document.addEventListener("DOMContentLoaded", () => {
  // Lấy các element từ DOM
  const form = document.getElementById("summarize-form");
  const inputText = document.getElementById("input-text");
  const submitBtn = document.getElementById("submit-btn");
  const loader = document.getElementById("loader");

  const beamWidthSlider = document.getElementById("beam-width");
  const beamWidthValueSpan = document.getElementById("beam-width-value");
  const tempSlider = document.getElementById("temperature");
  const tempValueSpan = document.getElementById("temperature-value");

  const resultsContainer = document.getElementById("results-container");
  const summaryOutput = document.getElementById("summary-output");
  const errorMessage = document.getElementById("error-message");

  // --- HÀM MỚI: Đồng bộ chiều cao của 2 ô văn bản ---
  const matchTextareaHeight = () => {
    // Đặt lại min-height để tính toán lại cho đúng khi resize
    summaryOutput.style.minHeight = "auto";
    // Lấy chiều cao thực tế của textarea
    const inputTextHeight = inputText.offsetHeight;
    // Gán chiều cao đó cho ô kết quả
    summaryOutput.style.minHeight = `${inputTextHeight}px`;
  };

  // Thiết lập placeholder ban đầu cho ô kết quả
  summaryOutput.textContent = "Kết quả tóm tắt sẽ xuất hiện ở đây...";
  summaryOutput.classList.add("placeholder");

  // Gọi hàm để đồng bộ chiều cao khi tải trang
  matchTextareaHeight();
  // Và gọi lại mỗi khi cửa sổ thay đổi kích thước
  window.addEventListener("resize", matchTextareaHeight);

  // Cập nhật giá trị hiển thị khi trượt slider
  beamWidthSlider.addEventListener("input", () => {
    beamWidthValueSpan.textContent = beamWidthSlider.value;
  });

  tempSlider.addEventListener("input", () => {
    tempValueSpan.textContent = parseFloat(tempSlider.value).toFixed(1);
  });

  // Xử lý khi submit form
  form.addEventListener("submit", async (event) => {
    event.preventDefault(); // Ngăn form gửi đi theo cách truyền thống

    const text = inputText.value.trim();
    if (!text) {
      showError("Vui lòng nhập văn bản cần tóm tắt.");
      return;
    }

    // Chuẩn bị cho việc gọi API
    setLoadingState(true);
    hideError();
    // resultsContainer.classList.add("hidden"); // Dòng này không cần thiết nữa

    try {
      const response = await fetch("/summarize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: text,
          beam_width: parseInt(beamWidthSlider.value, 10),
          temperature: parseFloat(tempSlider.value),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Lỗi HTTP ${response.status}`);
      }

      const data = await response.json();
      displaySummary(data.summary);
    } catch (error) {
      console.error("Lỗi khi tóm tắt:", error);
      showError(`Đã xảy ra lỗi: ${error.message}`);
    } finally {
      setLoadingState(false);
    }
  });

  function setLoadingState(isLoading) {
    if (isLoading) {
      submitBtn.disabled = true;
      loader.classList.remove("hidden");
    } else {
      submitBtn.disabled = false;
      loader.classList.add("hidden");
    }
  }

  function displaySummary(summaryText) {
    summaryOutput.textContent = summaryText;
    summaryOutput.classList.remove("placeholder"); // Xóa class placeholder khi có kết quả
    // resultsContainer.classList.remove("hidden"); // Dòng này không cần thiết nữa
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
  }

  function hideError() {
    errorMessage.classList.add("hidden");
  }
});
