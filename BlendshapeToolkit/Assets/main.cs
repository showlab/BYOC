using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using System;
using System.Data;
using System.Text;
using System.Threading;
using System.Linq;


public class main : MonoBehaviour
{
    private Button btn_Start;
    private Button btn_Next;
    private Button btn_Previous;
    private Button btn_PlayAnim; 
    private Button btn_PlayVideo;
    private Button btn_PlayBackVideo;


    private Button btn_Transition;
    private  static Image image;
    private Image anim_curve_bg;
    private Sprite sprite;
    private static Text label;
 
    private static List<Sprite> spriteList = new List<Sprite>();
    private static string[] files;
 
    private static int index = -1;
    DataTable dt = new DataTable();
    private List<string[]> bs_list = new List<string[]>();

    private List<string[]> bs_list_EMA = new List<string[]>();
    
    private List<string[]> pose_angle_list = new List<string[]>();

    private List<string[]> pose_angle_list_EMA = new List<string[]>();
    private SkinnedMeshRenderer avatar_bs;
    private Transform neck_pose_angle;

    
    private SkinnedMeshRenderer teeth_bs;

    private static bool is_play;
    private static bool is_play_back;


    // AnimationClip
    public AnimationClip clip_tran;
    public AnimationClip clip_exp_anim;
    public AnimationClip clip_neck_anim;

    public Animation anim;

    public Animation anim_teeth;
    public Animation anim_neck;
    public List<float> anim_curve = new List<float>();
    public List<RectTransform> dot_list = new List<RectTransform>();
    public List<int> keyframe_list = new List<int>();
    public Image dot; 
    public Button btn_dot; 
    public Image line;

    public Button btn_AddKeyframe;
    public Button btn_AdjustBlendshape;
    public Button btn_Apply;
    public Button btn_Export;
    public Button btn_Clear;



    public float origin_x;
    public float origin_y;

    public float bg_width;
    public float bg_height;

    public Text bs_value;
    public Text bs_target;

    public List<List<float>> pref_delta = new List<List<float>>();
    public List<List<int>> pref_frame = new List<List<int>>();

    public float[] pref_aver = new float[51];


    public String filePath = "./Human_adjusted_blendshapes.csv";


    void Start()
    {
        // Produce blendshape
        btn_Start = GameObject.Find("Canvas/AnimateButton").GetComponent<Button>();
        btn_Start.onClick.AddListener(AnimateBlendshape);

        // Next
        btn_Next = GameObject.Find("Canvas/NextButton").GetComponent<Button>();
        btn_Next.onClick.AddListener(NextAnimation);

        // Previous
        btn_Previous = GameObject.Find("Canvas/PreviousButton").GetComponent<Button>();
        btn_Previous.onClick.AddListener(PreviousAnimation);

        // Play Anim
        btn_PlayAnim = GameObject.Find("Canvas/AnimButton").GetComponent<Button>();
        btn_PlayAnim.onClick.AddListener(PlayAnim);

        //Play Video
        btn_PlayVideo = GameObject.Find("Canvas/VideoButton").GetComponent<Button>();
        btn_PlayVideo.onClick.AddListener(PlayVideo);

        //Play Back Video
        btn_PlayBackVideo = GameObject.Find("Canvas/BackVideoButton").GetComponent<Button>();
        btn_PlayBackVideo.onClick.AddListener(PlayBackVideo);

        // Transition
        btn_Transition = GameObject.Find("Canvas/TransitionButton").GetComponent<Button>();
        btn_Transition.onClick.AddListener(TransitionAnimation);

        // Add Keyframe
        btn_AddKeyframe = GameObject.Find("Canvas/KeyframeButton").GetComponent<Button>();
        btn_AddKeyframe.onClick.AddListener(AddKeyframe);

        // Adjust Blendshape
        btn_AdjustBlendshape = GameObject.Find("Canvas/BlendshapeButton").GetComponent<Button>();
        btn_AdjustBlendshape.onClick.AddListener(AdjustBlendshape);
        
        // Apply preferences
        btn_Apply = GameObject.Find("Canvas/ApplyButton").GetComponent<Button>();
        btn_Apply.onClick.AddListener(ApplyPreferences);

        // Export result
        btn_Export = GameObject.Find("Canvas/ExportButton").GetComponent<Button>();
        btn_Export.onClick.AddListener(ExportResults);

        // Clear adjusted blendshape
        btn_Clear = GameObject.Find("Canvas/ClearButton").GetComponent<Button>();
        btn_Clear.onClick.AddListener(ClearAdjustedBlendshape);

        // Image
        image = GameObject.Find("Canvas/Image").GetComponent<Image>();
        anim_curve_bg = GameObject.Find("Canvas/AnimationCurve").GetComponent<Image>();
        dot = GameObject.Find("Canvas/Dot").GetComponent<Image>();
        btn_dot = GameObject.Find("Canvas/DotButton").GetComponent<Button>();

        // Label
        label = GameObject.Find("Canvas/Text").GetComponent<Text>();

        // Avatar blendshape
        avatar_bs = GameObject.Find("blendshapes1").GetComponent<SkinnedMeshRenderer>();
        neck_pose_angle = GameObject.Find("Neck_Male").GetComponent<Transform>();

        // Teeth
        teeth_bs = GameObject.Find("Mouth").GetComponent<SkinnedMeshRenderer>();

        // Animator

        // Play video
        is_play = false;
        // Play back video
        is_play_back = false;


        clip_tran = new AnimationClip();
        clip_exp_anim = new AnimationClip();
        clip_neck_anim = new AnimationClip();


        anim = GameObject.Find("blendshapes1").GetComponent<Animation>();
        anim_teeth = GameObject.Find("Mouth").GetComponent<Animation>();
        anim_neck = GameObject.Find("Neck_Male").GetComponent<Animation>();



        origin_x = -1032f;
        origin_y = -572f;

        bg_width=2000f;
        bg_height=200f;


        line = GameObject.Find("Canvas/Line").GetComponent<Image>();

        bs_value = GameObject.Find("Canvas/ValueInput/Text").GetComponent<Text>();
        bs_target = GameObject.Find("Canvas/TargetInput/Text").GetComponent<Text>();


        for(int i=0;i<51;i++){
            List<float> arr = new List<float>();

            pref_delta.Add(arr);
        }

        for(int i=0;i<51;i++){
            List<int> arr = new List<int>();
            pref_frame.Add(arr);
        }


    }


    private void ClearAdjustedBlendshape(){
        for(int i=0;i<51;i++){
            pref_delta[i].Clear();
            pref_frame[i].Clear();
        }
        for(int i=0;i<files.Length;i++){
            Image keyPointImage = dot_list[i].GetComponent<Image>();
            keyPointImage.color = Color.white;
        }
        for(int k=0;k<keyframe_list.Count;k++){
            Image keyPointImage = dot_list[keyframe_list[k]].GetComponent<Image>();
            keyPointImage.color = Color.green;
        }
    }

    private void ApplyPreferences(){
        for(int i=1;i<51;i++){
            if(pref_delta[i].Count!=0){
                pref_aver[i]=pref_delta[i].Sum()/pref_delta[i].Count;
            }
        }
        for(int i=0;i<files.Length;i++){
            for(int j=1;j<51;j++){
                if(pref_frame[j].Count!=0 && ! pref_frame[j].Contains(i)){
                    bs_list[i][j] = (float.Parse(bs_list[i][j]) + pref_aver[j]).ToString();
                }
            }
        }

        ClearAdjustedBlendshape();


        for(int i=0;i<50;i++){
                avatar_bs.SetBlendShapeWeight(i,100*float.Parse(bs_list[index][i+1]));
        }

        teeth_bs.SetBlendShapeWeight(0,100*float.Parse(bs_list[index][22]));

        neck_pose_angle.localRotation = Quaternion.Euler(new Vector3(float.Parse(pose_angle_list[index][1]),
                                                                    float.Parse(pose_angle_list[index][2]),
                                                                    float.Parse(pose_angle_list[index][3])));
        for(int i=0;i<files.Length;i++){
            
            float sum = 0.0f;
            for(int j=0;j<50;j++){
                sum += float.Parse(bs_list[i][j+1]);
            }
            float avg = sum/50;
            anim_curve[i]=avg;
        }
        float min_value = anim_curve.Min();
        float max_value = anim_curve.Max();
        float k = (max_value-min_value)/bg_height;

        for(int i=0;i<files.Length;i++){
            float d = bg_width/files.Length;
            dot_list[i].anchoredPosition = new Vector2(origin_x+i*d, origin_y+(anim_curve[i]-min_value)/k);


        }


    }

    private void ExportResults(){
        using (StreamWriter sw = new StreamWriter(filePath)){
            foreach (string[] row in bs_list){
                string line = string.Join(",", row);
                
                sw.WriteLine(line);
            }
        }

    }

    private void AdjustBlendshape(){
        string value_str=bs_value.text;
        string target_str=bs_target.text;
        Debug.Log(value_str);
        Debug.Log(target_str);
        if(!string.IsNullOrEmpty(value_str) && !string.IsNullOrEmpty(target_str)){
            float value = float.Parse(value_str)*0.01f;
            int target = int.Parse(target_str);

            // preferences.Add(value);
            float delta = value - float.Parse(bs_list[index][target]);
            pref_frame[target].Add(index);
            pref_delta[target].Add(delta);

            // bs_list[index][target]=value_str;
            bs_list[index][target]=(float.Parse(value_str)*0.01f).ToString();

            float d = bg_width/files.Length;
            float sum = 0.0f;
            for(int j=0;j<50;j++){
                sum += float.Parse(bs_list[index][j+1]);
            }
            float avg = sum/50;
            anim_curve[index]=avg;
            
            float min_value = anim_curve.Min();
            float max_value = anim_curve.Max();
            float k = (max_value-min_value)/bg_height;
            dot_list[index].anchoredPosition = new Vector2(origin_x+index*d, origin_y+(anim_curve[index]-min_value)/k);


            Image keyPointImage = dot_list[index].GetComponent<Image>();
            keyPointImage.color = Color.red;

            

            for(int i=0;i<50;i++){
                avatar_bs.SetBlendShapeWeight(i,100*float.Parse(bs_list[index][i+1]));
            }
            teeth_bs.SetBlendShapeWeight(0,100*float.Parse(bs_list[index][22]));
            neck_pose_angle.localRotation = Quaternion.Euler(new Vector3(float.Parse(pose_angle_list[index][1]),
                                                                        float.Parse(pose_angle_list[index][2]),
                                                                        float.Parse(pose_angle_list[index][3])));

        }

    }

    private void AddKeyframe(){
        keyframe_list.Add(index);
        Image keyPointImage = dot_list[index].GetComponent<Image>();
        keyPointImage.color = Color.green;
    }

    private void TransitionAnimation(){
        AddBlendshapeKeyframe();

    }

    private void AddBlendshapeKeyframe(){
        AnimationCurve curve = new AnimationCurve();
        clip_tran.legacy=true;

        for(int i=0;i<50;i++){
            curve.AddKey(new Keyframe(0.0f, 0.0f));
            curve.AddKey(new Keyframe(1.5f, 100*float.Parse(bs_list[index][i+1])));
            curve.AddKey(new Keyframe(3.0f, 0.0f));
            clip_tran.SetCurve("", typeof(SkinnedMeshRenderer), "blendShape.shape_"+(i+1).ToString()+"_channel", curve);

            for(int j=0;j<3;j++){
                curve.RemoveKey(0);
            }
        
        }
        anim.AddClip(clip_tran,"Transition");

        curve.AddKey(new Keyframe(0.0f, 0.0f));
        curve.AddKey(new Keyframe(1.5f, 100*float.Parse(bs_list[index][22])));
        curve.AddKey(new Keyframe(3.0f, 0.0f));
        clip_tran.SetCurve("", typeof(SkinnedMeshRenderer), "blendShape.edit3_channel", curve);
        for(int j=0;j<3;j++){
            curve.RemoveKey(0);
        }

        anim_teeth.AddClip(clip_tran, "Transition_teeth");
        anim.Play("Transition");
        anim_teeth.Play("Transition_teeth");


    }

    private void PlayAnim(){
        if(files.Length==1){
            AddBlendshapeKeyframe();
            
        }
        else{
            AnimationCurve curve = new AnimationCurve();

            clip_exp_anim.legacy=true;
            clip_neck_anim.legacy=true;

            int frame_rate = 10;
            float duration = 1.0f/frame_rate;
            float cur_time = 0.0f;
            int cnt = 5;

            // Facial animation
            for(int i=0;i<50;i++){
                index=0;
                cur_time=0.0f;

                for(int k=0;k<keyframe_list.Count;k++){
                    cur_time = keyframe_list[k] * duration;
                    curve.AddKey(new Keyframe(cur_time, 100*float.Parse(bs_list[keyframe_list[k]][i+1])));

                }


                clip_exp_anim.SetCurve("", typeof(SkinnedMeshRenderer), "blendShape.shape_"+(i+1).ToString()+"_channel", curve);

                // Teeth animation
                if(i==21){
                    clip_exp_anim.SetCurve("", typeof(SkinnedMeshRenderer), "blendShape.edit3_channel", curve);
                }

                while(curve.length>=1){
                    curve.RemoveKey(0);
                }

            }

            // Neck pose animation
            index=0;
            cur_time=0.0f;
            AnimationCurve curve_x = new AnimationCurve();
            AnimationCurve curve_y = new AnimationCurve();
            AnimationCurve curve_z = new AnimationCurve();
            AnimationCurve curve_w = new AnimationCurve();


            while(index<files.Length){
                
                float euler_x = Quaternion.Euler(new Vector3(float.Parse(pose_angle_list[index][1]),
                                                            float.Parse(pose_angle_list[index][2]),
                                                            float.Parse(pose_angle_list[index][3])))[0];
                float euler_y = Quaternion.Euler(new Vector3(float.Parse(pose_angle_list[index][1]),
                                                            float.Parse(pose_angle_list[index][2]),
                                                            float.Parse(pose_angle_list[index][3])))[1];
                float euler_z = Quaternion.Euler(new Vector3(float.Parse(pose_angle_list[index][1]),
                                                            float.Parse(pose_angle_list[index][2]),
                                                            float.Parse(pose_angle_list[index][3])))[2];
                float euler_w = Quaternion.Euler(new Vector3(float.Parse(pose_angle_list[index][1]),
                                                            float.Parse(pose_angle_list[index][2]),
                                                            float.Parse(pose_angle_list[index][3])))[3];

                curve_x.AddKey(new Keyframe(cur_time, euler_x));
                curve_y.AddKey(new Keyframe(cur_time, euler_y));
                curve_z.AddKey(new Keyframe(cur_time, euler_z));
                curve_w.AddKey(new Keyframe(cur_time, euler_w));



                

                for(int j=0;j<cnt;j++){
                    cur_time+=duration;
                    index++;
                }
                
            }

            clip_neck_anim.SetCurve("", typeof(Transform), "m_LocalRotation.x", curve_x);
            clip_neck_anim.SetCurve("", typeof(Transform), "m_LocalRotation.y", curve_y);
            clip_neck_anim.SetCurve("", typeof(Transform), "m_LocalRotation.z", curve_z);
            clip_neck_anim.SetCurve("", typeof(Transform), "m_LocalRotation.w", curve_w);


            index=0;

            anim.AddClip(clip_exp_anim,"PlayAnim");
            anim_teeth.AddClip(clip_exp_anim,"PlayAnim_teeth");
            anim_neck.AddClip(clip_neck_anim,"PlayAnim_neck");

            anim.Play("PlayAnim");
            anim_teeth.Play("PlayAnim_teeth");
            anim_neck.Play("PlayAnim_neck");

            StartCoroutine(PlayImagesAndLabels());

        }

        

        
    }

    private void PlayVideo(){
        if(is_play==false){
            is_play=true;   
        }
        else{
            is_play=false;
        }
    }

    private void PlayBackVideo(){
        if(is_play_back==false){
            is_play_back=true;   
        }
        else{
            is_play_back=false;
        }
    }

    private void AnimateBlendshape()
    {
        // Image to blendshape
        Image2Blendshape();

        // Import all images
        GetSpriteList();
        Debug.Log("Import all images!");

        // Read csv
        ReadCsv();

        // Display label
        NextAnimation();

        // Draw an animation curve
        if(files.Length>1){
            DrawAnimationCurve();
        }

        // EMA for animation
        if(is_video()){
            Debug.Log("EMA for video");
            EMA();
        }

    }

    private bool is_video(){
        string strPath = @"../../image2bs/test_video"; 
        int f = Directory.GetFiles(strPath).Length; 
        if(f!=0){
            return true;
        }
        else{
            return false;
        }
    }

    // Exponential Moving Average
    private void EMA(){
        bs_list_EMA = bs_list;
        pose_angle_list_EMA = pose_angle_list;

        float beta_exp = 0.6f;

        for(int j=1;j<51;j++){
            for(int i=1;i<files.Length;i++){
                bs_list_EMA[i][j] = (beta_exp * float.Parse(bs_list_EMA[i-1][j]) + (1-beta_exp) * float.Parse(bs_list[i][j])).ToString();

            }
        }

        bs_list = bs_list_EMA;
        float beta_head = 0.8f;

        for(int j=1;j<4;j++){
            for(int i=1;i<files.Length;i++){
                pose_angle_list_EMA[i][j] = (beta_head * float.Parse(pose_angle_list_EMA[i-1][j]) + (1-beta_head) * float.Parse(pose_angle_list[i][j])).ToString();

            }
        }

        pose_angle_list = pose_angle_list_EMA;

    }

    private void DrawAnimationCurve(){
        Debug.Log("Draw an animation curve.");

        float d = bg_width/files.Length;

        for(int i=0;i<files.Length;i++){
            float sum = 0.0f;
            for(int j=0;j<50;j++){
                sum += float.Parse(bs_list[i][j+1]);
            }
            float avg = sum/50;
            anim_curve.Add(avg);

        }
        float min_value = anim_curve.Min();
        float max_value = anim_curve.Max();
        float k = (max_value-min_value)/bg_height;

        for(int i=0;i<files.Length;i++){
            DrawPoint(new Vector2(origin_x+i*d, origin_y+(anim_curve[i]-min_value)/k));



        }

        int cnt = 5;
        for(int i=0;i<files.Length;i+=cnt){
            keyframe_list.Add(i);
            Image keyPointImage = dot_list[i].GetComponent<Image>();
            keyPointImage.color = Color.green;
        }


    }

    private void DrawPoint(Vector2 position)
    {
        RectTransform pointTransform = Instantiate(dot.gameObject, position, Quaternion.identity, transform).GetComponent<RectTransform>();

        pointTransform.anchoredPosition = position;
        pointTransform.sizeDelta = new Vector2(7f, 7f);
        dot_list.Add(pointTransform);

    }

    private static void Image2Blendshape(){
        Debug.Log("Start to animate!");

        System.Diagnostics.Process proc = new System.Diagnostics.Process(); 
        proc.StartInfo.FileName = $@"../../image2bs/inference.bat";

        proc.StartInfo.UseShellExecute = true;
        proc.StartInfo.CreateNoWindow = false;
        proc.StartInfo.Verb = "runas";
        proc.Start();
        proc.WaitForExit();
        proc.Close();

        Debug.Log("End to animate!");
    }

    private void NextAnimation(){
        Debug.Log("Next picture");
        index++;
        if (index >= files.Length)
        {
            index = 0;
        }
        image.sprite = spriteList[index];
        label.text=(index+1).ToString()+" / "+files.Length.ToString();

        // Display blendshape
        for(int i=0;i<50;i++){
            avatar_bs.SetBlendShapeWeight(i,100*float.Parse(bs_list[index][i+1]));
        }

        teeth_bs.SetBlendShapeWeight(0,100*float.Parse(bs_list[index][22]));

        neck_pose_angle.localRotation = Quaternion.Euler(new Vector3(float.Parse(pose_angle_list[index][1]),
                                                                     float.Parse(pose_angle_list[index][2]),
                                                                     float.Parse(pose_angle_list[index][3])));


     }

    private void PreviousAnimation(){
        Debug.Log("Previous picture");
        index--;
        if (index < 0)
        {
            index = files.Length - 1;
        }
        image.sprite = spriteList[index];
        label.text=(index+1).ToString()+" / "+files.Length.ToString();

        // Display blendshape
        Debug.Log(bs_list[index][0]);
        for(int i=0;i<50;i++){
            avatar_bs.SetBlendShapeWeight(i,100*float.Parse(bs_list[index][i+1]));
        }

        teeth_bs.SetBlendShapeWeight(0,100*float.Parse(bs_list[index][22]));

        neck_pose_angle.localRotation = Quaternion.Euler(new Vector3(float.Parse(pose_angle_list[index][1]),
                                                                     float.Parse(pose_angle_list[index][2]),
                                                                     float.Parse(pose_angle_list[index][3])));


    }



    private void GetSpriteList()
    {
        files = Directory.GetFiles(@"../../image2bs/test_images");

        for (int i = 0; i < files.Length; i++)
        {
            if(files[i]=="detections"){
                continue;
            }
            spriteList.Add(TextureToSprite(LoadTextureByIO(files[i])));
        }
    }
 

    private Texture2D LoadTextureByIO(string path)
    {
        FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read);
        fs.Seek(0, SeekOrigin.Begin);
        byte[] bytes = new byte[fs.Length];

        try
        {
            fs.Read(bytes, 0, bytes.Length);
 
        }
        catch (Exception e)
        {
            Debug.Log(e);
        }
        fs.Close();
        // Debug.Log(bytes);
        int width = 700;
        int height = 700;
        Texture2D texture = new Texture2D(width, height);
        if (texture.LoadImage(bytes))
        {
            print("Images loaded.");
            return texture;
 
        }
        else
        {
            print("Images not loaded.");
            return null;
        }
    }
 
 
    private Sprite TextureToSprite(Texture2D tex)
    {
        Sprite sprite = Sprite.Create(tex, new Rect(0, 0, tex.width, tex.height), new Vector2(0.5f, 0.5f));
        return sprite;
    }

    private void ReadCsv(){
        FileStream fs = 
        File.OpenRead(@"../../image2bs/results/predicted_blendshape.csv");
        // File.OpenRead(@"./Human_adjusted_blendshapes.csv");

        StreamReader sr = new StreamReader(fs);

        string tempText = "";
        while ((tempText = sr.ReadLine()) != null && sr.ReadLine().Length != 1)
        {
            Debug.Log(tempText);
            
            string[] arr = tempText.Split(",".ToCharArray(), StringSplitOptions.RemoveEmptyEntries);
            Debug.Log(arr.Length);
            bs_list.Add(arr);
        }


        sr.Close(); 
        fs.Close();

        FileStream fs_ = 
        File.OpenRead(@"../../image2bs/results/predicted_pose_angle.csv");
        StreamReader sr_ = new StreamReader(fs_);

        string tempText_ = "";
        while ((tempText_ = sr_.ReadLine()) != null && sr_.ReadLine().Length != 1)
        {
            Debug.Log(tempText_);
            
            string[] arr = tempText_.Split(",".ToCharArray(), StringSplitOptions.RemoveEmptyEntries);
            Debug.Log(arr.Length);
            pose_angle_list.Add(arr);
        }

        sr_.Close(); 
        fs_.Close();

    }

    // Function for play video
    private void NextAnimation4Play(){
        Debug.Log("Next picture");
        index++;
        if (index >= files.Length)
        {
            index=0;
        }
        image.sprite = spriteList[index];
        label.text=(index+1).ToString()+" / "+files.Length.ToString();

        // Display blendshape
        Debug.Log(bs_list[index][0]);
        for(int i=0;i<50;i++){
            avatar_bs.SetBlendShapeWeight(i,100*float.Parse(bs_list[index][i+1]));
        }

        teeth_bs.SetBlendShapeWeight(0,100*float.Parse(bs_list[index][22]));

        neck_pose_angle.localRotation = Quaternion.Euler(new Vector3(float.Parse(pose_angle_list[index][1]),
                                                                     float.Parse(pose_angle_list[index][2]),
                                                                     float.Parse(pose_angle_list[index][3])));

        Thread.Sleep(50);


     }

     // Function for play back video
    private void NextAnimation4PlayBack(){
        Debug.Log("Next picture");
        index--;
        if (index < 0)
        {
            index=files.Length-1;
        }
        image.sprite = spriteList[index];
        label.text=(index+1).ToString()+" / "+files.Length.ToString();

        // Display blendshape
        Debug.Log(bs_list[index][0]);
        for(int i=0;i<50;i++){
            avatar_bs.SetBlendShapeWeight(i,100*float.Parse(bs_list[index][i+1]));
        }

        teeth_bs.SetBlendShapeWeight(0,100*float.Parse(bs_list[index][22]));

        neck_pose_angle.localRotation = Quaternion.Euler(new Vector3(float.Parse(pose_angle_list[index][1]),
                                                                     float.Parse(pose_angle_list[index][2]),
                                                                     float.Parse(pose_angle_list[index][3])));

        Thread.Sleep(50);


     }

     IEnumerator PlayImagesAndLabels(){
        while(index<files.Length){
            image.sprite = spriteList[index];
            label.text=(index+1).ToString()+" / "+files.Length.ToString();
            yield return new WaitForSeconds(0.1f);
            index++;

        }

        
     }



    // Update is called once per frame
    void Update()
    {
        if(index>-1&&index<files.Length&&files.Length>1){

            RectTransform lineTransform = line.GetComponent<RectTransform>();;
            lineTransform.position = new Vector3(dot_list[index].position[0], anim_curve_bg.transform.position[1],0.0f);
            lineTransform.sizeDelta = new Vector2(1f, 200f);
            Image lineImage = lineTransform.GetComponent<Image>();
            lineImage.color = Color.red;


        }

        if(is_play==true){
            // index=-1;
            NextAnimation4Play();
        }

        if(is_play_back==true){
            // index=-1;
            NextAnimation4PlayBack();
        }
        
    }
}
