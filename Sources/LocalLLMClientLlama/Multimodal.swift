import Foundation
import LocalLLMClientCore
@_exported import LocalLLMClientLlamaC

public class MultimodalContext: @unchecked Sendable {
    package let multimodalContext: OpaquePointer
    package let verbose: Bool

    package init(url: URL, context: Context, parameter: LlamaClient.Parameter) throws(LLMError) {
        var mparams = mtmd_context_params_default()
        mparams.use_gpu = true
        mparams.print_timings = parameter.options.verbose
        if let numberOfThreads = parameter.numberOfThreads {
            mparams.n_threads = Int32(numberOfThreads)
        }
        guard let multimodalContext = mtmd_init_from_file(url.path(percentEncoded: false), context.model.model, mparams) else {
            throw .failedToLoad(reason: "Failed to load the mmproj file")
        }
        self.multimodalContext = multimodalContext
        self.verbose = parameter.options.verbose
    }

    deinit {
        mtmd_free(multimodalContext)
    }

    package func chunks(images: [LLMInputImage]) throws(LLMError) -> MultimodalChunks {
        var bitmaps: [OpaquePointer?] = try images.map { image throws(LLMError) in
            let data = try llmInputImageToData(image)
            let (bytes, width, height) = imageDataToRGBBytes(imageData: data)!
            guard let bitmap = mtmd_bitmap_init(UInt32(width), UInt32(height), bytes) else {
                throw .failedToLoad(reason: "Failed to create bitmap")
            }
            return bitmap
        }
        defer {
            bitmaps.forEach(mtmd_bitmap_free)
        }

        let chunks = mtmd_input_chunks_init()!

        let textStorage = "    \(String(cString: mtmd_default_marker()))    " // spaces for the workaround of tokenizer

        // NeuraChat (fork): `mtmd_tokenize` se llama DENTRO de `withCString`.
        //
        // El puntero que entrega `withCString` solo es válido mientras dura el
        // closure. Antes se construía ahí el `mtmd_input_text` y se devolvía
        // FUERA, así que para cuando `mtmd_tokenize` leía `text.text` el puntero
        // ya colgaba: C no encontraba el marcador `<__media__>`, contaba cero
        // marcadores contra un bitmap y devolvía 1 → "Failed to tokenize bitmap".
        //
        // Es comportamiento indefinido, así que a veces "funciona" (la memoria
        // de pila sigue intacta por suerte) y a veces no. En un build Release
        // con -O para iOS fallaba SIEMPRE: la visión local no funcionaba nunca,
        // con cualquier modelo y cualquier imagen.
        // `bitmapCount` se saca FUERA a proposito. En la misma llamada convivian
        // `&bitmaps` (acceso de modificacion, porque el parametro C es
        // `const mtmd_bitmap **`: el const esta en el pointee, no en el puntero)
        // y `bitmaps.count` (lectura). A nivel de funcion Swift lo aceptaba, pero
        // dentro de un closure la variable pasa a estar CAPTURADA y la
        // exclusividad puede comprobarse en EJECUCION: seria cambiar un fallo
        // por un trap. Con el recuento ya resuelto, dentro del closure solo
        // queda un acceso y no hay solapamiento posible.
        let bitmapCount = bitmaps.count
        let status = textStorage.withCString { cString -> Int32 in
            var text = mtmd_input_text(text: cString, add_special: false, parse_special: true)
            return mtmd_tokenize(multimodalContext, chunks, &text, &bitmaps, bitmapCount)
        }

        guard status == 0 else {
            throw .failedToLoad(reason: "Failed to tokenize bitmap")
        }

        return MultimodalChunks(chunks: chunks)
    }
}

package final class MultimodalChunks: @unchecked Sendable {
    package let chunks: OpaquePointer

    public init(chunks: OpaquePointer) {
        self.chunks = chunks
    }

    deinit {
        mtmd_input_chunks_free(chunks)
    }
}

package extension Context {
    func decode(bitmap: MultimodalChunks, with multimodal: MultimodalContext) throws(LLMError) {
        var newPosition: Int32 = 0
        let chunk = mtmd_input_chunks_get(bitmap.chunks, 1) // 1: <space><img><space>

        let imageTokens = mtmd_input_chunk_get_tokens_image(chunk)

        if multimodal.verbose {
            llamaLog(level: .debug, message: "encoding image or slice...\n")
        }

        guard mtmd_encode(multimodal.multimodalContext, imageTokens) == 0 else {
            throw .failedToDecode(reason: "Failed to encode image")
        }

        let embd = mtmd_get_output_embd(multimodal.multimodalContext);
        guard mtmd_helper_decode_image_chunk(
            multimodal.multimodalContext,
            context,
            chunk,
            embd,
            position,
            0, // seq_id
            Int32(parameter.batch),
            &newPosition) == 0 else {
            throw .failedToDecode(reason: "Failed to decode image")
        }
    }
}
