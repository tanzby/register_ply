#include <iostream>
#include <thread>

#include <ceres/ceres.h>


#include <pcl/io/ply_io.h>
#include <pcl/octree/octree_pointcloud_occupancy.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;

using PointT = pcl::PointXYZ;


class Point2PointFactor
{
    const PointT& source;
    const PointT& target;

public:
    Point2PointFactor(const PointT& s, const PointT& t):source(s), target(t) {}

    template <typename T>
    bool operator()(const T * s, const T * q, const T *t, T *residuals) const
    {
        T scale = s[0];
        Eigen::Quaternion<T> q_eigen {q[ 3 ], q[ 0 ], q[ 1 ], q[ 2 ]};
        Eigen::Matrix<T, 3, 1> t_eigen {t[ 0 ], t[ 1 ], t[ 2 ]};

        Eigen::Matrix<T, 3, 1> p_s (T(source.x), T(source.y), T(source.z));
        Eigen::Matrix<T, 3, 1> p_t = scale * q_eigen.toRotationMatrix() * p_s + t_eigen;

        // dist cost
        residuals[0] = p_t(0, 0) - T(target.x);
        residuals[1] = p_t(1, 0) - T(target.y);
        residuals[2] = p_t(2, 0) - T(target.z);

        return true;
    }

    static ceres::CostFunction *Create(const PointT& s, const PointT& t)
    {
        return (new ceres::AutoDiffCostFunction<Point2PointFactor, 3, 1, 4, 3>(new Point2PointFactor(s, t)));
    }
};

void
estimate_normals(pcl::PointCloud<PointT>::Ptr cloud_in, pcl::PointCloud<pcl::Normal>::Ptr normals)
{

    pcl::NormalEstimationOMP<PointT, pcl::Normal> norm_est;
    norm_est.setNumberOfThreads(4);
    norm_est.setKSearch (10);
    norm_est.setInputCloud (cloud_in);
    norm_est.compute (*normals);
}

double
compute_cloud_resolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
    double resolution = 0.0;
    int numberOfPoints = 0;
    int nres;
    std::vector<int> indices(2);
    std::vector<float> squaredDistances(2);
    pcl::search::KdTree<PointT> tree;
    tree.setInputCloud(cloud);

    for (size_t i = 0; i < cloud->size(); ++i)
    {
        if (! pcl::isFinite(cloud->points[i]))
            continue;

        // Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
        if (nres == 2)
        {
            resolution += sqrt(squaredDistances[1]);
            ++numberOfPoints;
        }
    }
    if (numberOfPoints != 0)
        resolution /= numberOfPoints;

    return resolution;
}

void
compute_iss(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointCloud<PointT>::Ptr key_points)
{
    pcl::ISSKeypoint3D<PointT, PointT> iss_det;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

    double model_resolution = compute_cloud_resolution(cloud);

    cout<<"resolution: " << model_resolution << endl;

    iss_det.setMinNeighbors(10);
    iss_det.setThreshold21(0.975);
    iss_det.setThreshold32(0.975);
    iss_det.setNumberOfThreads(4);

    iss_det.setInputCloud(cloud);
    iss_det.setSearchMethod(tree);
    iss_det.setSalientRadius(6*model_resolution);   // 0.5
    iss_det.setNonMaxRadius(4*model_resolution);
    iss_det.compute(*key_points);

    cout << "iss points num: " << key_points->size() << endl;
}

void
compute_pfh(pcl::PointCloud<PointT>::Ptr keypoints, pcl::PointCloud<PointT>::Ptr cloud,
            pcl::PointCloud<pcl::Normal>::Ptr normals,pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors)
{

    pcl::FPFHEstimationOMP<PointT, pcl::Normal, pcl::FPFHSignature33> pfh;
    pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT> );
    pfh.setInputCloud(keypoints);
    pfh.setInputNormals(normals);
    pfh.setSearchSurface(cloud);
    pfh.setSearchMethod(kdtree);
    pfh.setRadiusSearch(0.5);
    pfh.compute(*descriptors);
    cout<<"Get pfh: "<<descriptors->points.size()<<endl;
}

void
find_match(pcl::PointCloud<pcl::FPFHSignature33>::Ptr model_descriptors,
           pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_descriptors, 
           pcl::CorrespondencesPtr model_scene_corrs)
{
    pcl::KdTreeFLANN<pcl::FPFHSignature33> matching;
    matching.setInputCloud(model_descriptors);

    for (size_t i = 0; i < target_descriptors->size(); ++i)
    {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);
        // Ignore NaNs.
        if (std::isfinite(target_descriptors->at(i).histogram[0]))
        {
            int neighborCount = matching.nearestKSearch(target_descriptors->at(i), 1, neighbors, squaredDistances);
            if (neighborCount == 1 && squaredDistances[0] < 0.1f)
            {
                pcl::Correspondence correspondence(neighbors[0], static_cast<int>(i), squaredDistances[0]);
                model_scene_corrs->push_back(correspondence);
                cout<<"( "<<correspondence.index_query<<","<<correspondence.index_match<<" )"<<endl;
            }
        }
    }
    std::cout << "Found " << model_scene_corrs->size() << " correspondences." << std::endl;
}

bool
estimation_pose(pcl::PointCloud<PointT>::Ptr source, pcl::PointCloud<PointT>::Ptr target,
               pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_descriptors,pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_descriptors,
               pcl::PointCloud<PointT>::Ptr alignedModel, Eigen::Matrix4d& out_result)
{
    pcl::SampleConsensusPrerejective<PointT, PointT, pcl::FPFHSignature33> pose;
    pose.setInputSource(source);
    pose.setInputTarget(target);
    pose.setSourceFeatures(source_descriptors);
    pose.setTargetFeatures(target_descriptors);
    pose.setCorrespondenceRandomness(10);
    pose.setInlierFraction(0.01f);
    pose.setNumberOfSamples(4);
    pose.setSimilarityThreshold(0.2f);
    pose.setMaxCorrespondenceDistance(15.0f);
    pose.setMaximumIterations(1000);

    pose.align(*alignedModel);

    if (pose.hasConverged())
    {
        Eigen::Matrix4f transformation = pose.getFinalTransformation();
        Eigen::Matrix3f rotation = transformation.block<3, 3>(0, 0);
        Eigen::Vector3f translation = transformation.block<3, 1>(0, 3);

        std::cout << "Transformation matrix:" << std::endl << std::endl;
        printf("\t\t    | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1), rotation(0, 2));
        printf("\t\tR = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1), rotation(1, 2));
        printf("\t\t    | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1), rotation(2, 2));
        std::cout << std::endl;
        printf("\t\tt = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1), translation(2));

        out_result = transformation.cast<double>();

        return true;
    }
    else std::cout << "Did not converge." << std::endl;
    return false;
}

void evaluation(pcl::PointCloud<pcl::PointXYZ>::Ptr source, pcl::PointCloud<pcl::PointXYZ>::Ptr target)
{
    pcl::octree::OctreePointCloudOccupancy<PointT> occupancy_checker(0.02);
    occupancy_checker.setOccupiedVoxelsAtPointsFromCloud(target);

    size_t is_in_counter = 0;

    for (size_t i = 0; i < source->points.size(); ++i)
    {
        if (occupancy_checker.isVoxelOccupiedAtPoint(source->points[i]))
        {
            is_in_counter++;
        }
    }

    cout << "total size of source: " << source->size() << endl;
    cout << "total size of target: " << target->size() << endl;
    cout << "hit count: " << is_in_counter << endl;
}

void transformPointCloudSim3(const pcl::PointCloud<PointT> &cloud_in,
                             pcl::PointCloud<PointT> &cloud_out, double scale, const Eigen::Vector3f &offset,
                             const Eigen::Quaternionf &rotation)
{
    pcl::transformPointCloud(cloud_in, cloud_out, {0, 0, 0}, rotation);
    for (auto&p: cloud_out.points)
    {
        p.x = scale * p.x + offset[0];
        p.y = scale * p.y + offset[1];
        p.z = scale * p.z + offset[2];
    }
}


int main(int argc, char** argv)
{
    if (argc != 3)
    {
        cerr << "./cloud_registration source.pcd target.pcd" << endl;
    }

    // read ply
    pcl::PointCloud<PointT>::Ptr source ( new pcl::PointCloud<PointT> );
    pcl::PointCloud<PointT>::Ptr target( new pcl::PointCloud<PointT> );
    pcl::PointCloud<PointT>::Ptr source_filtered ( new pcl::PointCloud<PointT> );
    pcl::PointCloud<PointT>::Ptr target_filtered( new pcl::PointCloud<PointT> );
    pcl::PointCloud<PointT>::Ptr source_transform ( new pcl::PointCloud<PointT> );
    pcl::PointCloud<PointT>::Ptr source_transform_filtered ( new pcl::PointCloud<PointT> );

    if (pcl::io::loadPLYFile(argv[1], *source) == -1)
    {
        cerr << "cannot load source" << endl;
        return 0;
    }
    else cout << "load " << source->size() <<" source points from file.\n";

    if (pcl::io::loadPLYFile(argv[2], *target) == -1)
    {
        cerr << "cannot load target" << endl;
        return 0;
    }
    else cout << "load " << target->size() <<" target points from file.\n";

    //downsampling
    pcl::VoxelGrid<PointT> voxel_filter;
    voxel_filter.setLeafSize(0.1, 0.1, 0.1);
    voxel_filter.setInputCloud(source);
    voxel_filter.filter(*source_filtered);

    voxel_filter.setLeafSize(0.05, 0.05, 0.05);
    voxel_filter.setInputCloud(target);
    voxel_filter.filter(*target_filtered);

    cout << "down-sample source points to " << source_filtered->size() << endl;
    cout << "down-sample target points to " << target_filtered->size() << endl;

    Eigen::Matrix4d init_transform;

    init_transform <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1;


    int max_iter_times = 50;

    pcl::registration::CorrespondenceEstimation<PointT, PointT> corr_estimator;
    corr_estimator.setInputTarget(target_filtered);


    pcl::visualization::PCLVisualizer viewer("Step by step icp");

    pcl::visualization::PointCloudColorHandlerCustom<PointT> source_cloud_color_handler(source_filtered, 255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> target_cloud_color_handler(target_filtered, 230, 20, 0);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> result_cloud_color_handler(source_filtered, 0, 255,0);

    viewer.addPointCloud(source_filtered, source_cloud_color_handler,  "source");
    viewer.addPointCloud(target_filtered, target_cloud_color_handler,  "target");

    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "result");

    mutex update_cloud_mutex;

    thread estimate_thread([&]()
    {

        Eigen::Matrix<double, 1, 1> scale; scale[0] = 0.5;
        Eigen::Vector3d t = init_transform.block<3, 1>(0, 3);
        Eigen::Quaterniond q (init_transform.topLeftCorner<3, 3>());


        // transform
        transformPointCloudSim3(*source_filtered, *source_transform_filtered, scale[0], t.cast<float>(), q.cast<float>());

        // compute normal
        pcl::PointCloud<pcl::Normal>::Ptr source_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::Normal>::Ptr target_normals(new pcl::PointCloud<pcl::Normal>);
        estimate_normals(source_transform_filtered, source_normals);
        estimate_normals(target_filtered, target_normals);

        // compute iss
        pcl::PointCloud<PointT>::Ptr source_iss_points(new pcl::PointCloud<PointT>);
        pcl::PointCloud<PointT>::Ptr target_iss_points(new pcl::PointCloud<PointT>);
        compute_iss(source_transform_filtered, source_iss_points);
        compute_iss(target_filtered, target_iss_points);

        {
            std::lock_guard<std::mutex>  lock(update_cloud_mutex);
            pcl::visualization::PointCloudColorHandlerCustom<PointT> model_keypoint_color(source_iss_points, 0, 0, 255);
            pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_keypoint_color(target_iss_points, 0, 0, 255);

            viewer.addPointCloud(source_iss_points, model_keypoint_color, "source_iss_points");
            viewer.addPointCloud(target_iss_points, scene_keypoint_color, "target_iss_points");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "source_iss_points");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "target_iss_points");
        }

        pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_descriptors_fpfh(new pcl::PointCloud<pcl::FPFHSignature33>());
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_descriptors_fpfh(new pcl::PointCloud<pcl::FPFHSignature33>());
        compute_pfh(source_iss_points, source_transform_filtered, source_normals, source_descriptors_fpfh);
        compute_pfh(target_iss_points, target_filtered, target_normals, target_descriptors_fpfh);

        pcl::PointCloud<PointT>::Ptr aligned_result(new pcl::PointCloud<PointT>);
        Eigen::Matrix4d  out_result;
        if(estimation_pose(source_iss_points, target_iss_points,
                           source_descriptors_fpfh, target_descriptors_fpfh, aligned_result, out_result))
        {
            t = out_result.block<3, 1>(0, 3);
            q = out_result.topLeftCorner<3, 3>();
        }

        // transform
        transformPointCloudSim3(*source_filtered, *source_transform_filtered, scale[0], t.cast<float>(), q.cast<float>());

        {
            std::lock_guard<std::mutex> lock(update_cloud_mutex);
            viewer.removePointCloud("result");
            viewer.addPointCloud(source_transform_filtered, result_cloud_color_handler,  "result");

            viewer.removePointCloud("source_iss_points");
            viewer.removePointCloud("target_iss_points");

        }


        for (int iter=0; iter < max_iter_times; ++iter)
        {
            cout << "\n\n ========================================= iter: "
                 << iter << " ========================================= \n\n";

            // transform
            pcl::transformPointCloud(*source_filtered, *source_transform_filtered, {0, 0, 0}, q.cast<float>());
            for (auto&p: source_transform_filtered->points)
            {
                p.x = scale[0] * p.x + t[0];
                p.y = scale[0] * p.y + t[1];
                p.z = scale[0] * p.z + t[2];
            }

            {
                std::lock_guard<std::mutex> lock(update_cloud_mutex);
                viewer.removePointCloud("result");
                viewer.addPointCloud(source_transform_filtered, result_cloud_color_handler,  "result");
            }

            // find correspondences
            corr_estimator.setInputSource(source_transform_filtered);
            pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
            corr_estimator.determineCorrespondences(*correspondences);

            // add residual block
            ceres::Problem problem;
            ceres::LocalParameterization * q_parameterization = new ceres::EigenQuaternionParameterization();

            problem.AddParameterBlock(scale.data(), 1);
            problem.AddParameterBlock(q.coeffs().data(), 4, q_parameterization);
            problem.AddParameterBlock(t.data(), 3);

            ceres::LossFunction* loss_function = new ceres::CauchyLoss(sqrt( 1.0 / correspondences->size()));
            if (iter < max_iter_times * 0.4) loss_function = nullptr;

            ceres::Problem::EvaluateOptions evaluate_options;
            for (auto& c: *correspondences)
            {
                ceres::CostFunction *cost_function;
                cost_function = Point2PointFactor::Create(source_filtered->points[c.index_query],
                                                          target_filtered->points[c.index_match]);

                auto res_id = problem.AddResidualBlock(cost_function, loss_function, scale.data(), q.coeffs().data(), t.data());

                if (iter == max_iter_times - 1) evaluate_options.residual_blocks.emplace_back(res_id);
            }

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = true;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            cout << summary.BriefReport() << endl;

            cout << "scale:" << scale << endl;
            cout << "R:\n" << q.toRotationMatrix() << endl;
            cout << "t:" << t.transpose() << endl;

            // reach the end of iteration
            if (iter == max_iter_times - 1)
            {
                transformPointCloudSim3(*source, *source_transform, scale[0], t.cast<float>(), q.cast<float>());
                evaluation(source_transform, target);
            }
        }

     }
    );

    while ( !viewer.wasStopped() )
    {
        {
            std::lock_guard<std::mutex> lock(update_cloud_mutex);
            viewer.spinOnce(10);
        }
    }

    return 0;
}
