use std::collections::HashMap;
use std::fs::File;
use std::ops::ControlFlow::Break;
use std::path::Path;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use plotters::backend::BitMapBackend;
use plotters::chart::{self, ChartBuilder};
use plotters::drawing::IntoDrawingArea;
use plotters::element::{Circle, PathElement};
use plotters::series::LineSeries;
use plotters::style::{Color, RGBColor, ShapeStyle};
use plotters::style::full_palette::{BLACK, BLUE, GREEN, RED, WHITE};
use rand::seq::SliceRandom;
use csv::ReaderBuilder;
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Dimension, Zip, s};
use ndarray_csv::Array2Reader;
use ndarray_rand::rand::thread_rng;

// use ndarray_rand::rand::{SeedableRng, rngs::StdRng};
// use ndarray_rand::rand_distr::Normal;
// use ndarray_rand::RandomExt;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::{self, SeedableRng};

fn sigmoid<S, D>(x: &ArrayBase<S, D>) -> Array<f32 , D>
where 
    S: ndarray::Data<Elem = f32>,
    D: Dimension,
{
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let mut exp_x = Array2::zeros(x.raw_dim()) ;
    // จัดการทีละแถว (Row) เพื่อทำ Softmax Stabilization
    for (i, row) in x.rows().into_iter().enumerate() {
        // หาค่ามากสุดในแถวนั้นๆ (เทียบเท่า x.max(1))
        let max_val = row.fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        // คำนวณ exp(x - max) ของแถวนั้น
        let mut exp_row = row.mapv(|v| (v - max_val).exp());
        // หาผลรวมของค่า exp ในแถว (เพื่อใช้เป็นตัวหาร)
        let sum_val = exp_row.sum();
        // หารทุกตัวในแถวด้วยผลรวม เพื่อแปลงให้เป็นความน่าจะเป็น (Probability)
        exp_row.mapv_inplace(|v| v / sum_val);
        // ยัดกลับเข้าแมทริกซ์ผลลัพธ์
        exp_x.row_mut(i).assign(&exp_row);
    }
    exp_x
}
fn ha_1hot(z: &Array1<f32>, n: usize) -> Array2<f32> {
    let rows = z.len();
    // สร้าง Matrix ขนาด (จำนวนข้อมูล, จำนวนคลาส) เริ่มต้นด้วย 0.0 ทั้งหมด
    let mut one_hot = Array2::<f32>::zeros((rows, n));
    for i in 0..rows {
        let class_idx = z[i] as usize;
        // ป้องกันกรณี index เกินขอบเขตของจำนวนคลาส n
        if class_idx < n {
            // จุดไหนที่ตรงกับคลาส ให้เปิดไฟเป็น 1.0
            one_hot[[i, class_idx]] = 1.0;
        }
    }
    one_hot
}
fn ha_entropy(z: &Array2<f32>, h: &Array2<f32>) -> f32 {
    let mut sum_loss = 0.0 ;
    let mut count = 0.0;
    let epsilon = 1e-10_f32;

    // ใช้ Zip วนลูปคู่กันระหว่างเวกเตอร์ z และ h เพื่อตรวจสอบเงื่อนไขทีละตำแหน่ง
    Zip::from(z).and(h).for_each(|&z_val, &h_val| {
        // เงื่อนไข z == 1 (เนื่องจากเป็น f32 จึงเช็คด้วยระยะห่างหรือเทียบใกล้เคียง)
        if (z_val - 1.0).abs() < 1e-5 {
            // สูตร: -( log(h + 1e-10) )
            sum_loss += -(h_val + epsilon).ln();
            count += 1.0;
        }
    });
    // หากลุ่มข้อมูล z == 1 เจอก็หาค่าเฉลี่ย (.mean) แต่ถ้าไม่มีเลยให้คืนค่า 0.0 เพื่อความปลอดภัย
    if count > 0.0 { sum_loss / count} else { 0.0 }
}

pub struct Perceptron {
    pub m: usize,// จำนวน Node ใน Hidden Layer
    pub eta: f32,// training Rate
    
    pub w1: Array2<f32>,// weight
    pub b1: Array1<f32>,//bias
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
    pub w3: Array2<f32>,
    pub b3: Array1<f32>,
    // เก็บประวัติ Loss
    pub entropy_train: Vec<f32>,
    pub entropy_val: Vec<f32>,
    // เก็บประวัติ Accuracy
    pub kernel_train: Vec<f32>,
    pub kernel_val: Vec<f32>,
    pub kernel_test: Vec<f32>,
}

impl  Perceptron {
    pub fn new(m: usize, eta: f32) -> Self {
        Perceptron {
            m,
            eta,
            w1: Array2::zeros((0,0)),
            b1: Array1::zeros(0),
            w2: Array2::zeros((0,0)),
            b2: Array1::zeros(0),
            w3: Array2::zeros((0,0)),
            b3: Array1::zeros(0),
            entropy_train: Vec::new(),
            entropy_val: Vec::new(),
            kernel_train: Vec::new(),
            kernel_val: Vec::new(),
            kernel_test: Vec::new(),
        }   
    }

    pub fn training(
        &mut self,
        x_train: &Array2<f32>,
        z_train: &Array1<f32>,
        x_val: &Array2<f32>,
        z_val: &Array1<f32>,
        x_test: &Array2<f32>,
        z_test: &Array1<f32>,
        loop_count: usize
    ) {
        let mut rng = StdRng::seed_from_u64(42);
        // หาจำนวนคลาส
        let max_class = z_train.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        let num_class = (max_class as usize) + 1;

        // แปลง Target เป็น One-Hot Matrix
        let z_train_one_hot = ha_1hot(z_train, num_class);
        let z_val_one_hot = ha_1hot(z_val, num_class);

        let input_dim = x_train.ncols();
        
        // let num_rows = x.nrows() as f32;
        
        // สุ่มเริ่มต้นค่าน้ำหนัก (Weights & Biases Initialization)
        // let normal_dist = Normal::new(0.0, 1.0).unwrap();
        // let w1_std = (2.0 / (x.ncols() + self.m) as f32).sqrt();
        // let w2_std = (2.0 / (self.m + num_class) as f32).sqrt();
        let fn_std_w1 = (2.0 / (input_dim + self.m) as f32).sqrt();
        let fn_std_w2 = (2.0 / (input_dim + self.m) as f32).sqrt();
        let fn_std_w3 = (2.0 / (self.m + num_class) as f32).sqrt();
        
        self.w1 = Array2::random_using((input_dim, self.m), Normal::new(0.0, fn_std_w1).unwrap(), &mut rng);
        self.b1 = Array1::zeros(self.m);
        self.w2 = Array2::random_using((input_dim, self.m), Normal::new(0.0, fn_std_w2).unwrap(), &mut rng);
        self.b2 = Array1::zeros(self.m);
        self.w3 = Array2::random_using((self.m, num_class), Normal::new(0.0, fn_std_w3).unwrap(), &mut rng);
        self.b3 = Array1::zeros(num_class);

        // self.entropy.clear();
        // self.kernel.clear();
        let n_train = x_train.nrows() as f32;
        let n_val = x_val.nrows() as f32;
        let n_test = x_test.nrows() as f32;
        
        'epoch: for i in 0..loop_count {
            // Feedforward
            let a1 = x_train.dot(&self.w1) +  &self.b1;
            let h1 = sigmoid(&a1);
            let a2 = x_train.dot(&self.w2) +  &self.b2;
            let h2 = sigmoid(&a2);
            let a3 = h1.dot(&self.w3) + h2.dot(&self.w3) + &self.b3;
            let h3 = softmax(&a3);

            // คำนวณ Loss (Entropy)
            let j_train = ha_entropy(&z_train_one_hot, &h3);
            let acc_train = self.calculate_accuracy(&h3, z_train);
            // self.entropy.push(j);

            // Backpropagation (คำนวณ Gradient)
            let ga3 = (&h3 - &z_train_one_hot) / n_train;
            let gh2 = ga3.dot(&self.w3.t());
            let ga2 = &gh2 * &h2 * &(1.0 - &h2);
            let gh1 = ga3.dot(&self.w3.t());
            let ga1 = &gh1 * &h1 * &(1.0 - &h1);
            
            // Gradient Descent ปรับค่าน้ำหนัก (Update Weights & Biases)
            self.w3 -= &(self.eta *  &(&h1 + &h2).t().dot(&ga3));
            self.b3 -= &(self.eta * &ga3.sum_axis(Axis(0)));
            self.w2 -= &(self.eta * &x_train.t().dot(&ga2));
            self.b2 -= &(self.eta * &ga2.sum_axis(Axis(0)));
            self.w1 -= &(self.eta * &x_train.t().dot(&ga1));
            self.b1 -= &(self.eta * &ga1.sum_axis(Axis(0)));

            // [VALIDATION SET] Evaluation
            let a1_val = x_val.dot(&self.w1) + &self.b1;
            let h1_val = sigmoid(&a1_val);
            let a2_val = x_val.dot(&self.w2) + &self.b2;
            let h2_val = sigmoid(&a2_val);
            let a3_val = h1_val.dot(&self.w3) + h2_val.dot(&self.w3) + &self.b3;
            let h3_val = softmax(&a3_val);

            let j_val = ha_entropy(&z_val_one_hot, &h3_val);
            let acc_val = self.calculate_accuracy(&h3_val, z_val);

            // [TEST SET] Evaluation
            let a1_test = x_test.dot(&self.w1) + &self.b1;
            let h1_test = sigmoid(&a1_test);
            let a2_test = x_test.dot(&self.w2) + &self.b2;
            let h2_test = sigmoid(&a2_test);
            let a3_test = h1_test.dot(&self.w3) + h2_test.dot(&self.w3) + &self.b3;
            let h3_test = softmax(&a3_test);

            let acc_test = self.calculate_accuracy(&h3_test, z_test);
            
            self.entropy_train.push(j_train);
            self.entropy_val.push(j_val);
            
            self.kernel_train.push(acc_train);
            self.kernel_val.push(acc_val);    
            self.kernel_test.push(acc_test); 
            
            if i % 100 == 99 {
                println!("epoch: {} | Train Acc: {:.3} | Val Acc: {:.3} | Test Acc: {:.3}", i + 1, acc_train, acc_val, acc_test);
            }

            if acc_train > 0.95 && acc_val > 0.95 {
                println!("stop at {} cuz Train , Validation Accuracy more than 95%", i + 1);
                break 'epoch;
            }
        }

    }

    fn calculate_accuracy(&self, preds: &Array2<f32>, targets: &Array1<f32>) -> f32 {
        let mut correct = 0.0;
        for i in 0..preds.nrows() {
            let row = preds.row(i);
            // หา argmax(1) ของแต่ละแถว
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for (idx, &v) in row.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_idx = idx;
                }
            }
            if max_idx == targets[i] as usize {
                correct +=  1.0;
            }
            
        }
        correct / (targets.len() as f32)
    }
    pub fn plot_result(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(filename, (1200, 500)).into_drawing_area();
        root.fill(&WHITE)?;
        let sub_areas = root.split_horizontally(600);
        // --- 1. พลอตกราฟ Loss ทางด้านซ้าย ---
        let area_loss = &sub_areas.0;
        let max_loss = self.entropy_train.iter().chain(&self.entropy_val).fold(f32::NEG_INFINITY, |m, &v| m.max(v)) * 1.1;
        let mut chart_loss = ChartBuilder::on(area_loss)
            .caption("loss history", ("sans-serif",20))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0..self.entropy_train.len(), 0.0_f32..max_loss)?;
        
        chart_loss.configure_mesh().disable_mesh().draw()?;
        
        chart_loss.draw_series(LineSeries::new(
            self.entropy_train.iter().enumerate().map(|(i, &v)| (i, v)), &GREEN))?
            .label("Train loss")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
        
        chart_loss.draw_series(LineSeries::new(
            self.entropy_val.iter().enumerate().map(|(i, &v)| (i, v)), &RED))?
            .label("Validation loss")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        chart_loss.configure_series_labels().background_style(&WHITE.mix(0.8)).draw()?;

        // --- 2. พลอตกราฟ Accuracy ทางด้านขวา ---
        let area_acc = &sub_areas.1;
        let mut chart_acc = ChartBuilder::on(area_acc)
            .caption("Accuracy history", ("sans-serif",20))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0..self.kernel_train.len(), 0.0_f32..1.05_f32)?;
        
        chart_acc.configure_mesh().disable_mesh().draw()?;
        
        chart_acc.draw_series(LineSeries::new(
            self.kernel_train.iter().enumerate().map(|(i, &v)| (i, v)), &GREEN))?
            .label("Train Accuracy")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
        
        chart_acc.draw_series(LineSeries::new(
            self.kernel_val.iter().enumerate().map(|(i, &v)| (i, v)), &RED))?
            .label("Validation Accuracy")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        
        chart_acc.draw_series(LineSeries::new(
            self.kernel_test.iter().enumerate().map(|(i, &v)| (i, v)), &BLUE))?
            .label("Test Accuracy")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
        
        chart_acc.configure_series_labels().background_style(&WHITE.mix(0.8)).draw()?;

        
        root.present()?;
        println!("save filed : '{}'", filename);
        Ok(())
    }
    
}

fn plot_intput_scatter (
    x_train: &Array2<f32>,
    z_train: &Array1<f32>,
    output_filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_filename, (700,700)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_max = x_train.column(0).fold(f32::NEG_INFINITY, |m, &v| m.max(v) ) * 1.1;
    let x_min = x_train.column(0).fold(f32::INFINITY, |m, &v| m.min(v)) * 0.9;
    let y_max = x_train.column(1).fold(f32::NEG_INFINITY, |m, &v| m.max(v)) * 1.1;
    let y_min = x_train.column(1).fold(f32::INFINITY, |m, &v| m.min(v)) * 0.9;

    
    let mut chart = ChartBuilder::on(&root)
        .caption("Input (train set)", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().disable_mesh().draw()?;

    for i in 0..x_train.nrows() {
        let px = x_train[[i, 0]];
        let py = x_train[[i, 1]];
        let class_val = z_train[i] as i32;

        let dot_color = match class_val {
            0 => RGBColor(255, 0, 0).mix(0.7),
            1 => RGBColor(0, 255, 0).mix(0.7),
            _ => RGBColor(0, 0,255).mix(0.7),
        };
        chart.draw_series(std::iter::once(Circle::new(
            (px, py),
            4,
            dot_color.filled(),
        )))?;

        chart.draw_series(std::iter::once(Circle::new(
            (px, py),
            4,
            ShapeStyle {
                    color: BLACK.to_rgba(),
                    filled: false,
                    stroke_width: 1,
                },
        )))?;
    }
    root.present()?;
    println!("Path file name: '{}'", output_filename);
    Ok(())
}


fn main() {
    // 1. ระบุชื่อไฟล์ตรงๆ เป็น .data
    let file_path = "data/iris.data";
    let file = File::open(file_path).expect("file data not fond");
    // จุดเปลี่ยนสำคัญ: ตั้งค่าเป็น false เพราะไฟล์ .data ส่วนใหญ่ไม่มีบรรทัดหัวข้อ (Header)
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);
    // 2. โหลดข้อมูลทั้งหมดเข้า Matrix (สมมติว่าคอลัมน์สุดท้ายเป็นตัวเลขคลาส 0.0, 1.0, 2.0 ไว้แล้ว)
    let raw_dataset: Array2<String> = reader
        .deserialize_array2_dynamic()
        .expect("failed to deserialize dataset");

    println!("load succusfully : {:?}", raw_dataset.dim());
    let num_rows = raw_dataset.nrows();

    let mut label_map: HashMap<String, f32> = HashMap::new();
    let mut next_id: f32 = 0.0;
    
    for i in 0..num_rows {
        let label = raw_dataset[[i, 4]].trim().to_string();
        if !label_map.contains_key(&label) {
            label_map.insert(label, next_id);
            next_id += 1.0;
        }
    }
    
    let mut x = Array2::<f32>::zeros((num_rows, 4));
    let mut z = Array1::<f32>::zeros(num_rows);

    for i in 0..num_rows {
        for j in 0..4 {
            x[[i, j]] = raw_dataset[[i, j]].trim().parse::<f32>().unwrap_or(0.0);
        }
        let label = raw_dataset[[i, 4]].trim();
        z[i] = *label_map.get(label).unwrap();
    }
    println!("found {} class: {:?}", label_map.len(), label_map);
    // println!("X: {:?}", x);
    // println!("Z: {:?}", z);

    let mut indices: Vec<usize>  = (0..num_rows).collect();
    let mut rng = rand::rng();
    indices.shuffle(&mut rng);
    
    let mut shuffled_x = Array2::<f32>::zeros(x.raw_dim());
    let mut shuffled_z = Array1::<f32>::zeros(z.raw_dim());
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        shuffled_x.row_mut(new_idx).assign(&x.row(old_idx));
        shuffled_z[new_idx] = z[old_idx];
    }

    let test_size = 25;
    let val_size = 25;
    let train_size = num_rows - test_size - val_size;

    let x_train = shuffled_x.slice(s![0..train_size, ..]).to_owned();
    let z_train = shuffled_z.slice(s![0..train_size]).to_owned();

    let x_val = shuffled_x.slice(s![train_size..(train_size + val_size), ..]).to_owned();
    let z_val = shuffled_z.slice(s![train_size..(train_size + val_size)]).to_owned();

    let x_test = shuffled_x.slice(s![(train_size + val_size).., ..]).to_owned();
    let z_test = shuffled_z.slice(s![(train_size + val_size)..]).to_owned();

    let mut model = Perceptron::new(4, 0.03);
    
    model.training(&x_train, &z_train, &x_val, &z_val, &x_test, &z_test, 5000);

    fn print_class_dist(name: &str, z: &Array1<f32>) {
        let mut counts = [0; 3];
        for &v in z.iter() {
            counts[v as usize] += 1;
        }
        println!("{}: setosa={}, versicolor={}, virginica={}", name, counts[0], counts[1], counts[2]);
    }
    
    print_class_dist("Train", &z_train);
    print_class_dist("Val", &z_val);
    print_class_dist("Test", &z_test);

    plot_intput_scatter(&x_train, &z_train, "graph.png").unwrap();
    model.plot_result("plot-graph.png").unwrap();
}