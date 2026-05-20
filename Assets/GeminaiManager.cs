using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using UnityEngine.EventSystems;
using System.Collections;
using TMPro;

public class GeminiManager : MonoBehaviour
{
    [SerializeField] private string apiKey = "여기에_API_KEY";

    [Header("UI")]
    [SerializeField] private TMP_Text responseText;
    [SerializeField] private TMP_Text countText;
    [SerializeField] private Button askButton;
    [SerializeField] private Button ttsButton;

    [Header("YOLO")]
    [SerializeField] private RunYOLO runYOLO;

    [Header("TTS")]
    [SerializeField] private TTSManager ttsManager;

    [Header("Prompt")]
    [SerializeField]
    private string systemPrompt =
        "너는 학교 조교야.";

    [SerializeField]
    private string userPrompt =
        "한국어로 상황을 100자 이내로 설명해줘. 앞으로 수업에 어떻게 쓸 물건들인지. 수업은 어떤 수업이 있을지 설명해줘.";

    private bool isProcessing = false;
    private int analysisCount = 0;

    private string lastAnswer = "";

    private const string API_URL =
        "https://generativelanguage.googleapis.com/v1beta/models/" +
        "gemini-3.5-flash:generateContent";

    void Start()
    {
        if (askButton != null)
        {
            askButton.onClick.AddListener(() => StartCoroutine(AskGemini()));
            AddButtonPressEffect(askButton);
        }

        if (ttsButton != null)
        {
            ttsButton.onClick.AddListener(PlayTTS);
            AddButtonPressEffect(ttsButton);
        }

        if (ttsManager == null)
        {
            ttsManager = GetComponent<TTSManager>();
        }
    }

    void AddButtonPressEffect(Button button)
    {
        EventTrigger trigger = button.gameObject.GetComponent<EventTrigger>();

        if (trigger == null)
            trigger = button.gameObject.AddComponent<EventTrigger>();

        EventTrigger.Entry pointerDown = new EventTrigger.Entry();
        pointerDown.eventID = EventTriggerType.PointerDown;
        pointerDown.callback.AddListener((_) => SetButtonAlpha(button, 0.4f));
        trigger.triggers.Add(pointerDown);

        EventTrigger.Entry pointerUp = new EventTrigger.Entry();
        pointerUp.eventID = EventTriggerType.PointerUp;
        pointerUp.callback.AddListener((_) => SetButtonAlpha(button, 1f));
        trigger.triggers.Add(pointerUp);

        EventTrigger.Entry pointerExit = new EventTrigger.Entry();
        pointerExit.eventID = EventTriggerType.PointerExit;
        pointerExit.callback.AddListener((_) => SetButtonAlpha(button, 1f));
        trigger.triggers.Add(pointerExit);
    }

    void SetButtonAlpha(Button button, float alpha)
    {
        if (button == null)
            return;

        CanvasGroup cg = button.GetComponent<CanvasGroup>();

        if (cg == null)
            cg = button.gameObject.AddComponent<CanvasGroup>();

        cg.alpha = alpha;
    }

    IEnumerator AskGemini()
    {
        if (isProcessing)
            yield break;

        isProcessing = true;

        if (responseText != null)
            responseText.text = "분석 중...";

        analysisCount++;

        if (countText != null)
        {
            countText.text = "분석횟수: " + analysisCount;
        }

        string label =
            runYOLO != null && !string.IsNullOrEmpty(runYOLO.lastDetectedLabel)
            ? runYOLO.lastDetectedLabel
            : "unknown";

        string dynamicPrompt =
            "감지된 물체: " + label + ". " + userPrompt;

        string json = "{" +
            "\"system_instruction\":{\"parts\":[{\"text\":\"" + EscapeJsonString(systemPrompt) + "\"}]}," +
            "\"contents\":[{\"parts\":[" +
            "{\"text\":\"" + EscapeJsonString(dynamicPrompt) + "\"}" +
            "]}]" +
            "}";

        using var req = new UnityWebRequest(API_URL + "?key=" + apiKey, "POST");

        req.uploadHandler = new UploadHandlerRaw(
            System.Text.Encoding.UTF8.GetBytes(json));

        req.downloadHandler = new DownloadHandlerBuffer();
        req.SetRequestHeader("Content-Type", "application/json");

        yield return req.SendWebRequest();

        if (req.result == UnityWebRequest.Result.Success)
        {
            GeminiResponse geminiResponse =
                JsonUtility.FromJson<GeminiResponse>(req.downloadHandler.text);

            if (geminiResponse != null &&
                geminiResponse.candidates != null &&
                geminiResponse.candidates.Length > 0 &&
                geminiResponse.candidates[0].content != null &&
                geminiResponse.candidates[0].content.parts != null &&
                geminiResponse.candidates[0].content.parts.Length > 0)
            {
                lastAnswer = geminiResponse.candidates[0].content.parts[0].text;

                if (responseText != null)
                    responseText.text = lastAnswer;
            }
            else
            {
                lastAnswer = "응답을 읽을 수 없습니다.";

                if (responseText != null)
                    responseText.text = lastAnswer;
            }
        }
        else
        {
            lastAnswer = "";

            if (responseText != null)
            {
                responseText.text =
                    "오류: " + req.error + "\n" + req.downloadHandler.text;
            }
        }

        isProcessing = false;
    }

    public void PlayTTS()
    {
        if (ttsManager == null)
        {
            Debug.LogWarning("TTSManager가 연결되지 않았습니다.");

            if (responseText != null)
                responseText.text = "TTSManager가 연결되지 않았습니다.";

            return;
        }

        string textToSpeak = "";

        if (!string.IsNullOrWhiteSpace(lastAnswer))
        {
            textToSpeak = lastAnswer;
        }
        else if (responseText != null && !string.IsNullOrWhiteSpace(responseText.text))
        {
            textToSpeak = responseText.text;
        }

        if (string.IsNullOrWhiteSpace(textToSpeak) ||
            textToSpeak == "분석 중...")
        {
            if (responseText != null)
                responseText.text = "먼저 분석하기 버튼을 눌러 답변을 받아주세요.";

            return;
        }

        ttsManager.Speak(textToSpeak);
    }

    string EscapeJsonString(string value)
    {
        if (string.IsNullOrEmpty(value))
            return "";

        return value
            .Replace("\\", "\\\\")
            .Replace("\"", "\\\"")
            .Replace("\n", "\\n")
            .Replace("\r", "\\r")
            .Replace("\t", "\\t");
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