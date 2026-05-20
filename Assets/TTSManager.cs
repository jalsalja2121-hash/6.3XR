using UnityEngine;

public class TTSManager : MonoBehaviour
{
    [Header("TTS Settings")]
    [SerializeField, Range(0.5f, 2.0f)]
    private float speechRate = 1.0f;

    [SerializeField, Range(0.5f, 2.0f)]
    private float pitch = 1.0f;

#if UNITY_ANDROID && !UNITY_EDITOR
    private AndroidJavaObject tts;
    private AndroidJavaObject activity;
#endif

    private bool isReady = false;
    private bool isInitializing = false;

    private const int SUCCESS = 0;
    private const int QUEUE_FLUSH = 0;
    private const int LANG_MISSING_DATA = -1;
    private const int LANG_NOT_SUPPORTED = -2;

    void Start()
    {
        InitTTS();
    }

    public void InitTTS()
    {
        if (isReady || isInitializing)
            return;

        isInitializing = true;

#if UNITY_ANDROID && !UNITY_EDITOR
        using (AndroidJavaClass unityPlayer =
               new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
        {
            activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity");
        }

        activity.Call("runOnUiThread", new AndroidJavaRunnable(() =>
        {
            tts = new AndroidJavaObject(
                "android.speech.tts.TextToSpeech",
                activity,
                new TTSInitListener(this)
            );
        }));
#else
        isReady = true;
        isInitializing = false;
        Debug.Log("Editor에서는 실제 TTS 대신 로그만 출력됩니다.");
#endif
    }

#if UNITY_ANDROID && !UNITY_EDITOR
    private void OnTTSInitialized(int status)
    {
        isInitializing = false;

        if (status != SUCCESS || tts == null)
        {
            isReady = false;
            Debug.LogError("TTS 초기화 실패");
            return;
        }

        using (AndroidJavaClass localeClass = new AndroidJavaClass("java.util.Locale"))
        {
            AndroidJavaObject koreanLocale =
                localeClass.GetStatic<AndroidJavaObject>("KOREAN");

            int result = tts.Call<int>("setLanguage", koreanLocale);

            if (result == LANG_MISSING_DATA || result == LANG_NOT_SUPPORTED)
            {
                isReady = false;
                Debug.LogWarning("한국어 TTS를 지원하지 않거나 한국어 음성 데이터가 없습니다.");
                return;
            }
        }

        tts.Call<int>("setSpeechRate", speechRate);
        tts.Call<int>("setPitch", pitch);

        isReady = true;
        Debug.Log("TTS 준비 완료");
    }
#endif

    public void Speak(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return;

        string cleanText = CleanText(text);

#if UNITY_ANDROID && !UNITY_EDITOR
        if (!isReady || tts == null)
        {
            Debug.LogWarning("TTS가 아직 준비되지 않았습니다.");
            return;
        }

        activity.Call("runOnUiThread", new AndroidJavaRunnable(() =>
        {
            if (tts != null)
            {
                tts.Call<int>(
                    "speak",
                    cleanText,
                    QUEUE_FLUSH,
                    null,
                    "gemini_tts"
                );
            }
        }));
#else
        Debug.Log("[TTS 재생 내용] " + cleanText);
#endif
    }

    public void Stop()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        if (tts == null || activity == null)
            return;

        activity.Call("runOnUiThread", new AndroidJavaRunnable(() =>
        {
            tts.Call<int>("stop");
        }));
#endif
    }

    string CleanText(string text)
    {
        return text
            .Replace("\n", " ")
            .Replace("\r", " ")
            .Replace("*", "")
            .Replace("#", "")
            .Replace("`", "")
            .Trim();
    }

    void OnDestroy()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        if (tts != null)
        {
            tts.Call<int>("stop");
            tts.Call("shutdown");
            tts.Dispose();
            tts = null;
        }

        if (activity != null)
        {
            activity.Dispose();
            activity = null;
        }
#endif
    }

#if UNITY_ANDROID && !UNITY_EDITOR
    private class TTSInitListener : AndroidJavaProxy
    {
        private readonly TTSManager manager;

        public TTSInitListener(TTSManager manager)
            : base("android.speech.tts.TextToSpeech$OnInitListener")
        {
            this.manager = manager;
        }

        private void onInit(int status)
        {
            manager.OnTTSInitialized(status);
        }
    }
#endif
}