use std::io::{self, Read};
use ssvm_tensorflow_interface;
use serde::Deserialize;

fn search_vec(vector: &Vec<f32>, labels: &Vec<&str>, value: &f32) -> (i32, String) {
    for (i, f) in vector.iter().enumerate() {
        if f == value {
            return (i as i32, labels[i].to_owned());
        }
    }
    return (-1, "Unclassified".to_owned());
}

fn main() {
    let model_data: &[u8] = include_bytes!("mobilenet_v2_1.4_224_frozen.pb");
    let labels = include_str!("imagenet_slim_labels.txt");
    let label_lines : Vec<&str> = labels.lines().collect();


    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer).expect("Error reading from STDIN");
    let obj: FaasInput = serde_json::from_str(&buffer).unwrap();
    let img_buf = base64::decode_config(&(obj.body), base64::STANDARD).unwrap();

    let flat_img = ssvm_tensorflow_interface::load_jpg_image_to_rgb32f(&img_buf, 224, 224);

    let mut session = ssvm_tensorflow_interface::Session::new(model_data, ssvm_tensorflow_interface::ModelType::TensorFlow);
    session.add_input("input", &flat_img, &[1, 224, 224, 3])
           .add_output("MobilenetV2/Predictions/Softmax")
           .run();
    let res_vec: Vec<f32> = session.get_output("MobilenetV2/Predictions/Softmax");
    let mut sorted_vec = res_vec.clone(); 
    sorted_vec.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let top1 = sorted_vec[0];
    let top2 = sorted_vec[1];
    let top3 = sorted_vec[2];

    let r1 = search_vec(&res_vec, &label_lines, &top1);
    let r2 = search_vec(&res_vec, &label_lines, &top2);
    let r3 = search_vec(&res_vec, &label_lines, &top3);

    println!("{}: {:.2}%\n{}: {:.2}%\n{}: {:.2}%"
        , r1.1, top1 * 100.0
        , r2.1, top2 * 100.0   
        , r3.1, top3 * 100.0   
        );
}

#[derive(Deserialize, Debug)]
struct FaasInput {
    body: String
}
