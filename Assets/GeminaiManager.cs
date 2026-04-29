using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using UnityEngine.EventSystems;
using System.Collections;
using TMPro;

public class GeminiManager : MonoBehaviour
{
    [SerializeField] private string apiKey = "여기에_API_KEY";
    [SerializeField] private TMP_Text responseText;
    [SerializeField] private TMP_Text countText;
    [SerializeField] private Button askButton;
    [SerializeField] private Camera arCamera;

    [SerializeField]
    private string systemPrompt =
        "너는 공장 안전 점검 전문가야. 항상 한국어로 두 문장 이내로 답해";
    [SerializeField]
    private string userPrompt =
        "지금 화면에 보이는 것의 안전 위험 요소를 말해줘";
    private bool isProcessing = false;
    private int analysisCount = 0;

    private const string API_URL =
        "https://generativelanguage.googleapis.com/v1beta/models/" +
        "gemini-2.5-flash-lite:generateContent";

    void Start()
    {
        askButton.onClick.AddListener(() => StartCoroutine(AskGemini()));

        // 버튼 누를 때 / 뗄 때 이벤트 등록
        EventTrigger trigger = askButton.gameObject.AddComponent<EventTrigger>();

        EventTrigger.Entry pointerDown = new EventTrigger.Entry();
        pointerDown.eventID = EventTriggerType.PointerDown;
        pointerDown.callback.AddListener((_) => SetButtonAlpha(0.4f)); // 흐리게
        trigger.triggers.Add(pointerDown);

        EventTrigger.Entry pointerUp = new EventTrigger.Entry();
        pointerUp.eventID = EventTriggerType.PointerUp;
        pointerUp.callback.AddListener((_) => SetButtonAlpha(1f)); // 원래대로
        trigger.triggers.Add(pointerUp);
    }

    void SetButtonAlpha(float alpha)
    {
        CanvasGroup cg = askButton.GetComponent<CanvasGroup>();
        if (cg == null) cg = askButton.gameObject.AddComponent<CanvasGroup>();
        cg.alpha = alpha;
    }

    IEnumerator AskGemini()
    {
        if (isProcessing) yield break;
        isProcessing = true;

        responseText.text = "분석 중...";

        // ———— 1단계: AR 카메라 프레임 캡처 ————
        RenderTexture rt = new RenderTexture(320, 240, 24);
        arCamera.targetTexture = rt;
        arCamera.Render();
        RenderTexture.active = rt;

        Texture2D screenshot = new Texture2D(320, 240);
        screenshot.ReadPixels(new Rect(0, 0, 320, 240), 0, 0);
        screenshot.Apply();

        arCamera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        // ———— 2단계: 이미지를 Base64로 변환 ————
        byte[] bytes = screenshot.EncodeToPNG();
        string base64 = System.Convert.ToBase64String(bytes);
        Destroy(screenshot);

        // ------ 3단계: Gemini API 요청 JSON 구성 ------
        // 이미지(inline_data)와 질문(text)을 함께 전송
        string json = "{" +
        "\"system_instruction\":{\"parts\":[{\"text\":\"" + systemPrompt + "\"}]}," +
        "\"contents\":[{\"parts\":[" +
        "{\"inline_data\":{\"mime_type\":\"image/png\",\"data\":\"" + base64 + "\"}}," +
        "{\"text\":\"" + userPrompt + "\"}" +
        "]}]}";

        // ———— 4단계: API 호출 ————
        using var req = new UnityWebRequest(API_URL + "?key=" + apiKey, "POST");
        req.uploadHandler = new UploadHandlerRaw(
            System.Text.Encoding.UTF8.GetBytes(json));
        req.downloadHandler = new DownloadHandlerBuffer();
        req.SetRequestHeader("Content-Type", "application/json");

        yield return req.SendWebRequest();

        // ———— 5단계: 응답 처리 ————
        if (req.result == UnityWebRequest.Result.Success)
        {
            analysisCount++;
            countText.text = $"Count: {analysisCount}";

            // JSON 응답에서 텍스트 부분만 추출 (간단 파싱)
            GeminiResponse geminiResponse =
                JsonUtility.FromJson<GeminiResponse>(req.downloadHandler.text);
            responseText.text = geminiResponse.candidates[0].content.parts[0].text;
        }
        else
        {
            responseText.text = "오류: " + req.error;
        }

        isProcessing = false;
    }
    [System.Serializable]
    public class GeminiResponse
    {
        public Candidate[] candidates;
    }

    [System.Serializable]
    public class Candidate
    {
        public Content content;
    }

    [System.Serializable]
    public class Content
    {
        public Part[] parts;
    }

    [System.Serializable]
    public class Part
    {
        public string text;
    }

}