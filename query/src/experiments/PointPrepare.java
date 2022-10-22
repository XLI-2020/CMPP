package experiments;

import algorithm.IDModel_SPQ;
import algorithm.Init;
import datagenerate.DataGen;
import indoor_entitity.Door;
import indoor_entitity.IndoorSpace;
import indoor_entitity.Partition;
import indoor_entitity.Point;
import utilities.DataGenConstant;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import static java.util.stream.Collectors.toList;

public class PointPrepare {
    public static String trajectorys = System.getProperty("user.dir") + "/data_trajectory/trajectory_";

    public static HashMap<Integer, HashMap<Integer, Point>> trajectory = new HashMap<>();
    public static int [] trajectory_time = new int[10];

    public static void trajectoryGen_save(String file_tra, String dataset, int num, int sampleInterval) throws IOException {
        genAllTrajectories(num, sampleInterval);
        String results = "";

        for (int i = 0; i < num; i++) {
            results += i;
            HashMap<Integer, Point> tra = trajectory.get(i);
            System.out.println("traId: " + i);
//            for (Integer t: tra.keySet()) {
//                results += "\t" + t;
//                Point p = tra.get(t);
//                System.out.println("t: " + t);
//                results += "," + p.getX() + "," + p.getY() + "," + p.getmFloor();
//            }
            for (int t = 0; t < 3600; t += sampleInterval) {
                results += "\t" + t;
                Point p = tra.get(t);
                System.out.println("t: " + t);
                results += "," + p.getX() + "," + p.getY() + "," + p.getmFloor();
            }
            results += "\n";
        }

        try {
            FileWriter fw = new FileWriter(file_tra + dataset);
            fw.write(results);
            fw.flush();
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

    }

    public static void trajectoryGen_read(String file_tra, String dataset, int sampleInterval) throws IOException {
        Path path1 = Paths.get(file_tra + dataset);
        Scanner scanner1 = new Scanner(path1);
        while (scanner1.hasNextLine()) {
            String line = scanner1.nextLine();
            String [] tempArr = line.split("\t");
            int traId = Integer.parseInt(tempArr[0]);
            HashMap<Integer, Point> tra = new HashMap<>();
            for (int i = 1; i < tempArr.length; i++) {
//                System.out.println("i: " + i);
                String [] pointArr = tempArr[i].split(",");
                int t = Integer.parseInt(pointArr[0]);
                double x = Double.parseDouble(pointArr[1]);
                double y = Double.parseDouble(pointArr[2]);
                int floor  = Integer.parseInt(pointArr[3]);
                tra.put(t, new Point(x, y, floor));

            }

            trajectory.put(traId, tra);
        }

    }

    public static void genAllTrajectories(int num, int sampleInterval) {
        int traId = 0;
        for (int i = 0; i < num; i++) {
            trajectory_time[i] = 0;
        }
        while (traId < num) {
            while (trajectory.get(traId) == null || trajectory_time[traId] < 3600) {
                int t = 0;
                if (trajectory_time[traId] > 0) {
                    t = trajectory_time[traId];
                }
                genTrajectory(t, traId, sampleInterval);
                genTrajectory_static(traId, sampleInterval);
            }
            traId++;
            System.out.println("traId: " + traId);
        }
    }

    public static void genTrajectory_static(int traId, int sampleInterval) {
//        int static_time = (int)(Math.random() * 500);
//        HashMap<Integer, Point> tra = trajectory.get(traId);
//        int original_time = trajectory_time[traId];
//        int t = original_time;
//        for (int i = 0; i < static_time / sampleInterval; i++) {
//            Point Point_prev = IndoorSpace.iPoint.get(IndoorSpace.iPoint.size() - 1);
//            IndoorSpace.iPoint.add(Point_prev);
//            t += sampleInterval;
//            tra.put(t, Point_prev);
//            trajectory_time[traId] = t;
//            System.out.println("t: " + t + ", x: " + Point_prev.getX() + ", y: " +  Point_prev.getY() + ", floor: " +  Point_prev.getmFloor());
//            trajectory.replace(traId, tra);
//        }
//        Point Point_prev = IndoorSpace.iPoint.get(IndoorSpace.iPoint.size() - 1);
//        IndoorSpace.iPoint.add(Point_prev);
//        t = original_time + static_time;
//        tra.put(t, Point_prev);
//        trajectory_time[traId] = t;
//        System.out.println("t: " + t + ", x: " + Point_prev.getX() + ", y: " +  Point_prev.getY() + ", floor: " +  Point_prev.getmFloor());
//        trajectory.replace(traId, tra);

        int num = (int)(Math.random() * 5);
        for (int i = 0; i < num; i++) {
            Point Point_prev = IndoorSpace.iPoint.get(IndoorSpace.iPoint.size() - 1);
            IndoorSpace.iPoint.add(Point_prev);
            HashMap<Integer, Point> tra = trajectory.get(traId);
            tra = trajectory.get(traId);
            int t = tra.size() * sampleInterval;
            tra.put(t, Point_prev);
            System.out.println("t: " + t + ", x: " + Point_prev.getX() + ", y: " +  Point_prev.getY() + ", floor: " +  Point_prev.getmFloor());
            trajectory.replace(traId, tra);
        }

    }
    public static void genTrajectory(int t, int traId, int sampleInterval) {
        Point point1 = pickPoint();
        if (t > 0) {
            point1 = IndoorSpace.iPoint.get(IndoorSpace.iPoint.size() - 1);
        }
        String path = pickAnotherPoint(point1, (int)(Math.random() * 500));
        String[] pathArr = path.split("\t");
        System.out.println("pathArr.length: " + pathArr.length);

        // sampling
        Point p = point1;
        int parId = locPartition(p);
        IndoorSpace.iPoint.add(p);
        HashMap<Integer, Point> tra = new HashMap<>();
        if (trajectory.get(traId) != null) {
            tra = trajectory.get(traId);
        }
        tra.put(t, p);
        if (trajectory.get(traId) != null) {
            trajectory.replace(traId, tra);
        }
        else {
            trajectory.put(traId, tra);
        }
        double sampleDist = sampleInterval * DataGenConstant.traveling_speed;
//        System.out.println("path 0: " + pathArr[0]);
//        System.out.println("path 1: " + pathArr[1]);
//        System.out.println("path 2: " + pathArr[2]);

        for (int i = 2; i < pathArr.length - 1; i++) {
//            System.out.println("start sampling...");
            Door p2 = IndoorSpace.iDoors.get(Integer.parseInt(pathArr[i]));
            double p2pDist = p.eDist(p2);
            if (i > 2) {
                ArrayList<Integer> list1 = IndoorSpace.iDoors.get(Integer.parseInt(pathArr[i - 1])).getmPartitions();
                ArrayList<Integer> list2 = p2.getmPartitions();
                List<Integer> intersection = list1.stream().filter(item -> list2.contains(item)).collect(toList());
                if (intersection.size() == 1) {
                    parId = intersection.get(0);
//                    System.out.println("parId: " + parId);
                }
                else {
                    System.out.println("something wrong with TrajectoryGen.genTrajectory(");
                }
//
            }
            Partition par = IndoorSpace.iPartitions.get(parId);
            while (p2pDist > sampleDist) {
                p = sample(p, p2, p2pDist, sampleDist);
                IndoorSpace.iPoint.add(p);
                tra.put(t, p);
                trajectory_time[traId] = t;
                System.out.println("tra size: " + tra.size());
                if (trajectory.get(traId) != null) {
                    trajectory.replace(traId, tra);
                }
                else {
                    trajectory.put(traId, tra);
                }
                t += sampleInterval;
//                if (t >= 1000) break;
                p2pDist = p.eDist(p2);
                sampleDist = sampleInterval * DataGenConstant.traveling_speed;
            }
            if (p2pDist == sampleDist) {
                p = p2;
                IndoorSpace.iPoint.add(p);
                tra.put(t, p);
                trajectory_time[traId] = t;
                if (trajectory.get(traId) != null) {
                    trajectory.replace(traId, tra);
                }
                else {
                    trajectory.put(traId, tra);
                }
                t += sampleInterval;
//                if (t >= 1000) break;
                sampleDist = sampleInterval * DataGenConstant.traveling_speed;
            }
            else {
                p = p2;
                sampleDist = sampleInterval * DataGenConstant.traveling_speed - p2pDist;
            }
        }



    }

    public static Point sample(Point p1, Point p2, double p2pDist, double sampleDist) {
        double xDist = (sampleDist) * Math.abs(p2.getX() - p1.getX()) / p2pDist;
        double yDist = (sampleDist) * Math.abs(p2.getY() - p1.getY()) / p2pDist;
        double x = 0;
        double y = 0;
        int floor = p1.getmFloor();
        if (p1.getX() >= p2.getX()) {
            x = p1.getX() - xDist;
        }
        else {
            x = p1.getX() + xDist;
        }
        if (p1.getY() >= p2.getY()) {
            y = p1.getY() - yDist;
        }
        else {
            y = p1.getY() + yDist;
        }
        return new Point(x, y, floor);
    }

    public static String pickAnotherPoint(Point point1, double distance) {
        String result = "";

        IDModel_SPQ idModel_spq = new IDModel_SPQ();


        while (true) {
            Point point2 = pickPoint();
            result = idModel_spq.pt2ptDistance3(point1, point2);
            //            System.out.println("point2 is finished");
            double dist = Double.parseDouble(result.split("\t")[0]);
            //            System.out.println(3);
            //            double dist = Double.parseDouble(distStr.split("\t")[0]);
            if (dist > distance) {
                return point2.getX() + "," + point2.getY() + "," + point2.getmFloor() + "\t" + result;
            }
        }

    }

    public static Point pickPoint() {
        Point point = null;
        while (true) {
            int x = (int) (Math.random() * (2100 + 1));
            int y = (int) (Math.random() * (2700 + 1));
            int floor = 0;
            if (isLegal(x, y, floor)) {
                point = new Point(x, y, floor);
                System.out.println("point: " + point.getX() + "," + point.getY() + "," + point.getmFloor());
                return point;
            }
        }
    }

    /**
     * check whether the object is legal
     *
     * @param x
     * @param y
     * @param floor
     * @return
     */
    public static boolean isLegal(double x, double y, int floor) {
        if (floor > DataGenConstant.nFloor) {
            System.out.println("something wrong with the random floor");
            return false;
        }

        ArrayList<Integer> pars = IndoorSpace.iFloors.get(floor).getmPartitions();
        for (int i = 0; i < pars.size(); i++) {
            Partition par = IndoorSpace.iPartitions.get(pars.get(i));
            if (x >= par.getX1() && x <= par.getX2() && y >= par.getY1() && y <= par.getY2()) {
                System.out.println("parId: " + par.getmID());
                return true;
            }
        }

        return false;
    }

    public static int getHostPartition(Point point) {
        int partitionId = -1;
        int floor = point.getmFloor();
        ArrayList<Integer> pars = IndoorSpace.iFloors.get(floor).getmPartitions();
        for (int i = 0; i < pars.size(); i++) {
            Partition par = IndoorSpace.iPartitions.get(pars.get(i));
            if (point.getX() >= par.getX1() && point.getX() <= par.getX2() && point.getY() >= par.getY1() && point.getY() <= par.getY2()) {
                partitionId = par.getmID();
                return partitionId;
            }
        }
        return partitionId;
    }

    /**
     * locate a partition according to location string
     *
     * @param point
     * @return partition
     */
    public static int locPartition(Point point) {
        int partitionId = -1;

        int floor = point.getmFloor();
        ArrayList<Integer> pars = IndoorSpace.iFloors.get(floor).getmPartitions();
        for (int i = 0; i < pars.size(); i++) {
            Partition par = IndoorSpace.iPartitions.get(pars.get(i));
            if (point.getX() >= par.getX1() && point.getX() <= par.getX2() && point.getY() >= par.getY1() && point.getY() <= par.getY2()) {
                partitionId = par.getmID();
                return partitionId;
            }
        }

        return partitionId;
    }

    public static void main(String [] arg) throws IOException {
        Init.init("pre", 60);

        trajectoryGen_save(trajectorys, "newHSM", 10, 60);
//        trajectoryGen_read(trajectorys, "newHSM", 60);
//        for (int i = 0; i < 10; i++) {
//            String result = "" + i;
//            System.out.println("i: " + i);
//            for (int t = 0; t < DataGenConstant.endTime; t += 60) {
//                Point p = trajectory.get(i).get(t);
//                int partitionId = getHostPartition(p);
//
//                result = "\t" + t + "," + p.getX() + "," + p.getY() + "," + p.getmFloor();
//                System.out.println(result);
//                System.out.println("partition: " + partitionId);
//                if (partitionId == -1) {
//                    System.out.println("........................................................");
//                }
//
//
//            }
//
//        }

//        for (int i = 0; i < 1; i++) {
//            Trajectory object = IndoorSpace.iObject.get(i);
//            HashMap<Integer, ArrayList<Integer>> trajectory = object.getTrajectory();
//            ArrayList<Integer> times = object.getTime();
//            for (int t: times) {
//
//                System.out.println("t" + t + ", trajectory: " + trajectory.get(t));
//            }
//        }
//
//
//        for (int i = 0; i < IndoorSpace.iTraPoint.size(); i++) {
//            TrajectoryPoint trajectoryPoint = IndoorSpace.iTraPoint.get(i);
//            double x = trajectoryPoint.getX();
//            double y = trajectoryPoint.getY();
//            int floor = trajectoryPoint.getmFloor();
//            int parId = trajectoryPoint.getParID();
//            int objectId = trajectoryPoint.getObjectID();
//            double probability = trajectoryPoint.getProbability();
//            int t = trajectoryPoint.getT();
//
//            System.out.println("x: " + x + ", y: " + y + ", floor: " + floor + ", parId: " + parId + ", objectId: " + objectId + ", probability: " + probability + ", t: " + t);
//
//        }
//
//        DataGen dataGen = new DataGen();
//        dataGen.drawFloor();
    }
}
