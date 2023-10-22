use onnxruntime::ndarray::ArrayView1;

use plotters::chart::ChartBuilder;

use plotters::prelude::*;

use plotters::coord::Shift;

pub fn plot_charts<'a, DB: DrawingBackend>(
    results: impl Iterator<Item = ArrayView1<'a, f32>>,
    root: &DrawingArea<DB, Shift>,
) {
    const FRAME_TIME_MS: u32 = 160;

    let mut chart = ChartBuilder::on(&root)
        .caption("vad", ("Arial", 50).into_font())
        .margin(5)
        .x_label_area_size(50)
        .y_label_area_size(30)
        .build_cartesian_2d(0..(293u32 * FRAME_TIME_MS), 0f32..2f32)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    let colors = [RED, BLUE, MAGENTA, CYAN, BLACK];

    results
        .zip(colors.iter().cloned().cycle())
        .enumerate()
        .for_each(|(chart_n, (v, color))| {
            chart
                .draw_series(LineSeries::new(
                    v.iter()
                        .enumerate()
                        .map(|(i, y)| ((i * 160).try_into().unwrap(), *y)),
                    &color,
                ))
                .unwrap()
                .label(format!("speaker {chart_n}"))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        });

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();
}

//pub fn plot_series<'a, DB: DrawingBackend + Ranged, CT: CoordTranslate + Ranged>(
//    series: ArrayView1<'a, f32>,
//    style: impl Into<ShapeStyle> + Clone + 'static,
//    label: impl Into<String>,
//    mut chart: ChartContext<DB, Cartesian2d<DB, CT>>,
//) where
//    for<'b> &'b DynElement<'static, DB, (f32, f32)>: PointCollection<
//        'b,
//        (
//            <DB as plotters::prelude::Ranged>::ValueType,
//            <CT as plotters::prelude::Ranged>::ValueType,
//        ),
//    >,
//{
//    chart
//        .draw_series(LineSeries::new(
//            series.iter().enumerate().map(|(i, y)| (i as f32, *y)),
//            style.clone(),
//        ))
//        .unwrap()
//        .label(label)
//        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style.clone()));
//}
