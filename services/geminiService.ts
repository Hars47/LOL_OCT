
import { GoogleGenAI, Type } from "@google/genai";
import type { AnalysisResult } from '../types';

const fileToGenerativePart = async (file: File) => {
  const base64EncodedDataPromise = new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
    reader.readAsDataURL(file);
  });
  return {
    inlineData: { data: await base64EncodedDataPromise, mimeType: file.type },
  };
};

const parseApiError = (error: unknown): string => {
    if (error instanceof Error) {
        const message = error.message.toLowerCase();
        if (message.includes('api key not valid')) {
            return 'Invalid API Key. Please ensure your API key is correctly configured in your environment variables. You can verify your key on the Google AI Studio dashboard.';
        }
        if (message.includes('quota') || message.includes('rate limit')) {
            return 'API Quota Exceeded. You have made too many requests in a short period. Please wait a moment before trying again or check your usage limits in the Google Cloud console.';
        }
        if (message.includes('blocked') || message.includes('safety')) {
            return 'Content Safety Error. The request was blocked due to safety settings, which can occasionally be triggered by medical images. Please try a different image or adjust safety settings if possible.';
        }
        if (message.includes('server error') || message.includes('500')) {
             return 'AI Service Unavailable. The service is currently experiencing issues on the backend. Please try again in a few minutes.';
        }
    }
    // Generic fallback
    return 'An unexpected error occurred during the analysis. Please check your network connection and try again. If the problem persists, check the developer console for more details.';
}

export const analyzeImage = async (imageFile: File, refinementFeedback?: string): Promise<{ analysis: AnalysisResult, segmentedImageBase64?: string, heatmapImageBase64: string, segmentationUncertaintyMapBase64?: string }> => {
  if (!process.env.API_KEY) {
    throw new Error("API Key Not Found: The API_KEY environment variable is not set. Please configure it before running the application.");
  }
  
  try {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    const imagePart = await fileToGenerativePart(imageFile);

    // --- Task 1 (Parallel): Initiate all API calls concurrently ---
    
    // Segmentation & Uncertainty Maps (only on first run)
    const segmentationPromise = refinementFeedback ? Promise.resolve(null) : ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: {
          parts: [
              imagePart,
              { text: `Generate a medical segmentation map from this retinal OCT scan. Use a distinct, high-contrast color palette to clearly delineate different retinal layers and pathological features. It is critical that you follow these color assignments for pathologies:
- **Intraretinal and Subretinal Fluid:** Use shades of **vibrant blue** to color any fluid-filled spaces.
- **Drusen/Deposits:** Use shades of **bright yellow** to highlight any drusen or sub-RPE deposits.
- **Disorganized Tissue/CNV:** Use shades of **red** to indicate areas of choroidal neovascularization.
- **Healthy Retinal Layers:** Use other contrasting colors like green, teal, and magenta for the different healthy retinal layers.
**Crucially, you must embed a clear, readable text legend directly onto the bottom of the output image that explains the color mapping.** For example: "Color Key: Blue=Fluid, Yellow=Deposits/Drusen, Red=CNV, Green/Teal=Retinal Layers". The legend text should be white or another high-contrast color against a dark bar for maximum readability.`}
          ]
      },
      config: {
          responseModalities: ['IMAGE'],
      }
    });
    
    const segmentationUncertaintyPromise = refinementFeedback ? Promise.resolve(null) : ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: {
            parts: [
                imagePart,
                { text: `
                  **Task:** Generate a segmentation uncertainty map for the provided retinal OCT scan.

                  **Objective:** The map must visually represent the AI's confidence in its segmentation of different regions, highlighting areas where the segmentation is less reliable.

                  **Mandatory Instructions:**
                  1.  **Color Scale:** You MUST use a cool-to-warm color scale. For example, from dark purple/blue (cool) to bright yellow/white (warm).
                  2.  **Color Mapping:**
                      -   **High Confidence (Low Uncertainty):** Represent these areas with **cool, dark colors** (e.g., dark purple, deep blue). These are regions with clear, well-defined features like healthy retinal layers.
                      -   **High Uncertainty (Low Confidence):** Represent these areas with **warm, bright colors** (e.g., bright yellow, white). This is the most critical part. You must highlight the following:
                          - The precise, fuzzy edges of any fluid pockets (DME/CNV).
                          - The indistinct boundaries of drusen deposits.
                          - Any ambiguous or blurred borders between retinal layers.
                          - Any regions affected by imaging artifacts, noise, or low signal quality.
                  3.  **Output Format:** The final output must be a heatmap-style image overlaid on the original scan structure, where the color intensity directly corresponds to the level of uncertainty.`
                }
            ]
        },
        config: {
            responseModalities: ['IMAGE'],
        }
    });

    // Classification and Explanation
    const classificationSchema = {
      type: Type.OBJECT,
      properties: {
          diagnosis: {
              type: Type.STRING,
              description: "The most likely diagnosis. Must be one of: 'AMD', 'CNV', 'DME', 'Drusen', 'Normal', 'Geographic Atrophy'.",
          },
          confidence: {
              type: Type.STRING,
              description: "A confidence score for the diagnosis, as a percentage string (e.g., '95.7%').",
          },
          explanation: {
              type: Type.STRING,
              description: "A detailed clinical description of the findings in the OCT image that support the diagnosis.",
          },
          explainability: {
              type: Type.STRING,
              description: "Describe how a multi-task hybrid model analyzes this image. Explain the roles of: 1) Deeplab's atrous spatial pyramid pooling for multi-scale features. 2) TransUNet's transformers for global reasoning. 3) An attention mechanism that fuses these features, allowing the model to focus on salient pathology. 4) A multi-task framework that learns segmentation and classification concurrently. 5) Relate this to the 'Attention Heatmap' visualization, explaining what the highlighted areas signify for the final diagnosis.",
          },
          uncertaintyStatement: {
              type: Type.STRING,
              description: "An assessment of diagnostic uncertainty. Describe any factors that make the diagnosis challenging (e.g., poor image quality, subtle features, feature overlap between conditions) or state that confidence is high if features are unambiguous.",
          },
          segmentationUncertaintyStatement: {
              type: Type.STRING,
              description: "A brief explanation of uncertainty in the segmentation map. Describe regions with ambiguous boundaries, like fluid edges or indistinct layers, or state confidence is high.",
          },
          anomalyReport: {
              type: Type.STRING,
              description: "An optional report on any secondary, ancillary findings or anomalies detected that are not part of the primary diagnosis (e.g., 'epiretinal membrane noted'). If no anomalies, this field can be omitted.",
          }
      },
      required: ["diagnosis", "confidence", "explanation", "explainability", "uncertaintyStatement", "segmentationUncertaintyStatement"]
    };
    
    let classificationPrompt = `
      You are an expert AI ophthalmologist. Analyze the provided retinal OCT image with extreme precision. Your task is to classify it as Age-related Macular Degeneration (AMD), Geographic Atrophy, Choroidal Neovascularization (CNV), Diabetic Macular Edema (DME), Drusen, or Normal.

      **Clinical Criteria for Classification:**
      - **AMD (Age-related Macular Degeneration):** A spectrum of disease.
          - **Dry AMD:** Characterized by **large, soft drusen**, RPE changes (hyperpigmentation or hypopigmentation). Does NOT involve fluid.
          - **Wet AMD:** Characterized by the presence of **CNV**. This is the neovascular form of AMD.
      - **Geographic Atrophy (GA):** An advanced form of Dry AMD. Characterized by **well-demarcated areas of RPE and outer retinal thinning/loss**. If this is the primary finding, diagnose as GA.
      - **DME (Diabetic Macular Edema):** Characterized by retinal swelling from **leaking macular blood vessels** in patients with diabetic retinopathy. Your primary evidence is the presence of **intraretinal fluid** (dark, cyst-like spaces within the retina) and/or **subretinal fluid** without a clear neovascular membrane.
      - **Drusen:** Can be a standalone finding or part of AMD. Characterized by **solid waste material** as distinct bumps under the RPE. If drusen are large, confluent, and accompanied by RPE changes, the diagnosis should be AMD.
      - **CNV (Choroidal Neovascularization):** This is the hallmark of Wet AMD. It is caused by **abnormal blood vessel growth from the choroid** that penetrates the RPE layer. A definitive diagnosis requires a forensic, evidence-based approach. You must identify a combination of the following signs:
        1.  **The Core Lesion:** Locate the source of the problem—a **disruptive, disorganized, often hyper-reflective lesion** under or breaking through the RPE. This represents the neovascular membrane itself.
        2.  **Associated Fluid Leakage:** The abnormal vessels are leaky. Therefore, you must find associated **subretinal fluid** (lifting the retina) and/or **intraretinal fluid** (causing cystic swelling).
      - **Normal:** Shows well-defined retinal layers, a clear foveal depression, and a complete absence of the pathological signs listed above.

      **CRITICAL DIFFERENTIATION: FLUID vs. DEPOSITS (NON-NEGOTIABLE)**
      - **Fluid (Indicative of DME/CNV):** Visually appears as **dark, optically empty, cyst-like pockets** within or beneath the retina. Its presence causes **measurable retinal thickening, swelling, and separation of layers**. Fluid is the hallmark of active, sight-threatening disease.
      - **Deposits (Indicative of Drusen/AMD):** Visually appear as **solid, lumpy, often reflective accumulations** of material located beneath the RPE layer. They are deposits, NOT fluid pockets.

      **MANDATORY DIAGNOSTIC HIERARCHY (Diagnostic Funnel):**
      You MUST follow this reasoning process. Your highest priority is detecting fluid.
      1.  **STEP 1: CHECK FOR FLUID.** Is there any intraretinal or subretinal fluid?
          - **YES -> The scan is 'WET'.** The diagnosis MUST be either **CNV (Wet AMD)** or **DME**.
              - **Crucial Rule:** A diagnosis of 'Drusen' or 'Dry AMD' is strictly incorrect if any fluid is present. Drusen can coexist, but the primary diagnosis must be the fluid-related condition.
              - **Differentiating DME vs. CNV (Wet AMD):** To differentiate, your primary task is to **locate the source of the fluid.**
                  - **Hunt for the Neovascular Lesion:** Meticulously search for the hallmark of CNV: a **disruptive, fibrovascular lesion** under or through the RPE. If this lesion is identified as the source of the fluid, the diagnosis is unequivocally **CNV (Wet AMD)**.
                  - **Diagnose DME by Exclusion:** If, and only if, you observe fluid (especially cystic intraretinal fluid) **WITHOUT** being able to identify a definitive underlying neovascular membrane, should you diagnose **DME**. This implies the fluid is from diabetic-related microvascular leakage.
          - **NO -> The scan is 'DRY'.** The diagnosis must be **Geographic Atrophy**, **Dry AMD**, **Drusen**, or **Normal**.
              - **Crucial Rule:** A diagnosis of 'DME' or 'CNV' is impossible without fluid.
              - **Prioritize GA Detection:** First, you MUST check for Geographic Atrophy. If you identify a **well-demarcated zone of RPE and outer retinal thinning or loss**, your primary diagnosis MUST be **Geographic Atrophy**.
              - **Then Assess for Dry AMD:** Only if GA is absent should you then consider a diagnosis of **Dry AMD**, which requires the presence of large/confluent drusen and significant RPE changes.
              - **Then Drusen:** If only a few small/medium drusen are present without other signs of AMD, diagnose as **Drusen**.
              - **Finally, Normal:** If the retina is clear of all these signs, diagnose as **Normal**.

      **Secondary Anomaly Scan:**
      After establishing the primary diagnosis, perform a final check for any other anomalies not covered by the main diagnosis (e.g., epiretinal membrane, vitreomacular traction, lamellar hole). If found, describe them in the 'anomalyReport' field.

      **Uncertainty Assessment:**
      Provide a qualitative assessment of your diagnostic certainty. Note any ambiguities like poor image quality, subtle features, or overlapping signs.

      **Segmentation Uncertainty Assessment:**
      Briefly describe areas where segmentation would be challenging (e.g., indistinct boundaries, noise).

      Your response must be a JSON object conforming to the provided schema.
    `;

    if (refinementFeedback) {
        classificationPrompt += `\n\nA previous analysis was performed. The user has provided the following feedback to refine your diagnosis: "${refinementFeedback}". Please re-evaluate the image, taking this crucial feedback into account. Adjust your diagnosis, confidence, and explanations accordingly.`;
    }

    const classificationPromise = ai.models.generateContent({
      model: 'gemini-2.5-pro',
      contents: { parts: [imagePart, {text: classificationPrompt}]},
      config: {
          responseMimeType: "application/json",
          responseSchema: classificationSchema,
      }
    });

    // Attention Heatmap
    let heatmapPrompt = `Generate a visual attention map for this retinal OCT scan. Overlay a heatmap on the original image, using warm colors (like red and yellow) to highlight the most pathologically significant regions that would be influential for a diagnosis. Focus on features like fluid pockets, drusen deposits, or areas of retinal thinning. The rest of the image should be slightly desaturated to make the heatmap stand out.`;
    
    if (refinementFeedback) {
        heatmapPrompt = `A previous analysis was performed on this OCT scan. The user has provided feedback: "${refinementFeedback}". Generate a NEW attention heatmap that specifically focuses on the areas relevant to the user's feedback. The heatmap should reflect a re-evaluation of the image based on this new input. Use warm colors (red, yellow) for important areas and desaturate the background.`;
    }

    const heatmapPromise = ai.models.generateContent({
        model: 'gemini-2.5-flash-image',
        contents: { parts: [imagePart, { text: heatmapPrompt }] },
        config: {
            responseModalities: ['IMAGE'],
        }
    });

    // --- Task 2: Await all parallel promises ---
    const [
      segmentationResponse, 
      segmentationUncertaintyResponse, 
      classificationResponse, 
      heatmapResponse
    ] = await Promise.all([
      segmentationPromise, 
      segmentationUncertaintyPromise, 
      classificationPromise,
      heatmapPromise
    ]);
    
    // --- Task 3: Process all results ---
    
    // Process Classification Response
    const classificationText = classificationResponse.text.trim();
    let analysisResult: AnalysisResult;
    try {
      analysisResult = JSON.parse(classificationText);
    } catch (e) {
      console.error("Failed to parse JSON response:", classificationText);
      throw new Error("Could not parse the analysis result from the AI. The format was invalid.");
    }
    
    // Check confidence threshold and override diagnosis if necessary.
    const confidenceValue = parseFloat(analysisResult.confidence);
    const CONFIDENCE_THRESHOLD = 70;
    
    if (!isNaN(confidenceValue) && confidenceValue < CONFIDENCE_THRESHOLD) {
        const originalDiagnosis = analysisResult.diagnosis;
        analysisResult.diagnosis = 'Requires Further Review';
        analysisResult.uncertaintyStatement = `**Low Confidence Flag:** The AI's confidence of ${analysisResult.confidence} is below the ${CONFIDENCE_THRESHOLD}% threshold. The initial finding was **'${originalDiagnosis}'**. This result is highly uncertain and requires careful review. ${analysisResult.uncertaintyStatement}`;
    }
    
    // Process Heatmap Response
    const heatmapImagePart = heatmapResponse.candidates?.[0]?.content?.parts?.[0];
    if (!heatmapImagePart || !('inlineData' in heatmapImagePart) || !heatmapImagePart.inlineData) {
      throw new Error("Failed to generate heatmap image.");
    }
    const heatmapImageBase64 = heatmapImagePart.inlineData.data;
    
    // During a refinement, segmentation responses are null, so we only return the new analysis and heatmap.
    if (!segmentationResponse || !segmentationUncertaintyResponse) {
      return { analysis: analysisResult, heatmapImageBase64 };
    }

    // For an initial analysis, extract the new segmentation and uncertainty maps.
    const segmentedImagePart = segmentationResponse.candidates?.[0]?.content?.parts?.[0];
    if (!segmentedImagePart || !('inlineData' in segmentedImagePart) || !segmentedImagePart.inlineData) {
      throw new Error("Failed to generate segmented image.");
    }
    const segmentedImageBase64 = segmentedImagePart.inlineData.data;

    const segmentationUncertaintyMapPart = segmentationUncertaintyResponse.candidates?.[0]?.content?.parts?.[0];
    if (!segmentationUncertaintyMapPart || !('inlineData' in segmentationUncertaintyMapPart) || !segmentationUncertaintyMapPart.inlineData) {
      throw new Error("Failed to generate segmentation uncertainty map.");
    }
    const segmentationUncertaintyMapBase64 = segmentationUncertaintyMapPart.inlineData.data;

    return { analysis: analysisResult, segmentedImageBase64, heatmapImageBase64, segmentationUncertaintyMapBase64 };

  } catch (err) {
      console.error("Gemini API Error:", err);
      const userFriendlyMessage = parseApiError(err);
      throw new Error(userFriendlyMessage);
  }
};
