/**
 * 
 */
package indoor_entitity;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;

import utilities.DataGenConstant;

/**
 * <h1>Door</h1>
 * to describe a door
 *
 */
public class Door extends Point implements Comparable<Object>{
	private int mID;		// the door ID
	
	private ArrayList<Integer> mPartitions = new ArrayList<Integer>();	// the accessible partitions
	
	private ArrayList<Integer> mFloors = new ArrayList<Integer>();	// the accessible floors
	
	private ArrayList<Direction> mDirections = new ArrayList<Direction>();		// the direction of the door

	private HashMap<String, Double> lamda = new HashMap<>();

	private int timeInterval;

	private HashMap<Integer, HashMap<String, Double>> flows = new HashMap<>(); // key is the sampling time, value is numbers of objects through this door at that sampling time

	private HashMap<Integer, HashMap<String, Double>> preFlows = new HashMap<>();

	private double oTime;
	private double cTime;
	private int sType;	// door type: public or private
	
	/**
	 * Constructor
	 * 
	 * @param x
	 * @param y
	 * @param mType
	 */
	public Door(double x, double y, int mType) {
		super(x, y, DataGenConstant.mID_Floor, mType);
		this.mID = DataGenConstant.mID_Door++;
		this.oTime = 0;
		this.cTime = 24.00;
	}
	
	public Door(double x, double y, int mType, double oTime, double cTime) {
		super(x, y, DataGenConstant.mID_Floor, mType);
		this.mID = DataGenConstant.mID_Door++;
		this.oTime = oTime;
		this.cTime = cTime;
	}

	public Door(double x, double y) {
		super(x, y);
		this.mID = DataGenConstant.mID_Door++;
	}
	
	/**
	 * Constructor
	 * 
	 * @param another is a point
	 */
	public Door(Point another) {
		super(another);
		this.setmType(another.getmType());
	}
	
	/**
	 * add a accessible partition for this door
	 * 
	 * @param parID
	 */
	public void addPar(int parID) {
		if (this.mPartitions != null) {
			if (!this.mPartitions.contains(parID)) {
				this.mPartitions.add(parID);
			}
		}
	}
	
	/**
	 * add a accessible floor for this door
	 * 
	 * @param floorID
	 */
	public void addFloor(int floorID) {
		if (!this.mFloors.contains(floorID)) {
			this.mFloors.add(floorID);
		}
	}

	public void setLamda(String s, double lamda) {
		if (this.lamda.get(s) == null) {
			this.lamda.put(s, lamda);
		}
		else {
			this.lamda.replace(s, lamda);
		}
	}

	public double getLamda(String s) {
		return this.lamda.get(s);
	}

	public HashMap<String, Double> getLamda() {
		return this.lamda;
	}

	public void setTimeInterval(int interval) {
		this.timeInterval = interval;
	}

	public int getTimeInterval() {
		return this.timeInterval;
	}

	/**
	 * add flow numbers to the flow
	 * @param sampleTime
	 * @param s
	 * @param num
	 */
	public void setFlow(int sampleTime, String s, double num) {
		if (flows.get(sampleTime) == null) {
			HashMap<String, Double> flowInfo = new HashMap<>();
			flowInfo.put(s, num);
			flows.put(sampleTime, flowInfo);
		}
		else if (flows.get(sampleTime).get(s) == null) {
			HashMap<String, Double> flowInfo = flows.get(sampleTime);
			flowInfo.put(s, num);
			flows.replace(sampleTime, flowInfo);
		}
		else {
			HashMap<String, Double> flowInfo = flows.get(sampleTime);
			flowInfo.replace(s, num);

		}
	}

	public double getFlow(int sampleTime, String s) {
		if (flows.get(sampleTime) == null) {
			return -1;
		}
		else if (flows.get(sampleTime).get(s) == null) {
			return -1;
		}
		else {
			return flows.get(sampleTime).get(s);
		}
	}

	public HashMap<Integer, HashMap<String, Double>> getFlows() {
		return this.flows;
	}

	public void setFlows(HashMap<Integer, HashMap<String, Double>> flows) {
		this.flows = flows;
	}

	/**
	 * add the flow number to the preFlow
	 * @param sampleTime
	 * @param s
	 * @param num
	 */
	public void addPreFlow(int sampleTime, String s, double num) {
		if (preFlows.get(sampleTime) == null) {
			HashMap<String, Double> flowInfo = new HashMap<>();
			flowInfo.put(s, num);
			preFlows.put(sampleTime, flowInfo);

		}
		else if (preFlows.get(sampleTime).get(s) == null) {
			HashMap<String, Double> flowInfo = preFlows.get(sampleTime);
			flowInfo.put(s, num);
			preFlows.replace(sampleTime, flowInfo);
		}
		else {
			HashMap<String, Double> flowInfo = preFlows.get(sampleTime);
			flowInfo.replace(s, num);
		}
	}

	public double getPreFlow(int sampleTime, String s) {
		if (preFlows.get(sampleTime) == null) {
			return -1;
		}
		else if (preFlows.get(sampleTime).get(s) == null) {
			return -1;
		}
		else {
			return this.preFlows.get(sampleTime).get(s);
		}
	}

	public void clearPreFlows() {
		this.preFlows.clear();
	}

	/**
	 * @return the mID
	 */
	public int getmID() {
		return mID;
	}

	/**
	 * @param mID
	 *            the mID to set
	 */
	public void setmID(int mID) {
		this.mID = mID;
	}
	
	/**
	 * @return the oTime
	 */
	public double getoTime() {
		return oTime;
	}

	/**
	 * @param oTime
	 *            the oTime to set
	 */
	public void setoTime(double oTime) {
		this.oTime = oTime;
	}

	/**
	 * @return the eTime
	 */
	public double getcTime() {
		return cTime;
	}

	/**
	 * @param eTime
	 *            the eTime to set
	 */
	public void setcTime(double eTime) {
		this.cTime = eTime;
	}

	/**
	 * @return the sType
	 */
	public int getsType() {
		return sType;
	}

	/**
	 * @param sType
	 *            the sType to set
	 */
	public void setsType(int sType) {
		this.sType = sType;
	}

	/**
	 * @return the mPartitions
	 */
	public ArrayList<Integer> getmPartitions() {
		return mPartitions;
	}
	
	/**
	 * @param mPartitions
	 *            the mPartitions to set
	 */
	public void setmPartitions(ArrayList<Integer> mPartitions) {
		this.mPartitions = mPartitions;
	}
	
	/**
	 * @return the mFloors
	 */
	public ArrayList<Integer> getmFloors() {
		return mFloors;
	}
	
	/**
	 * @param mFloors
	 *            the mFloors to set
	 */
	public void setmFloors(ArrayList<Integer> mFloors) {
		this.mFloors = mFloors;
	}
	
	/**
	 * @param direction
	 *            the direction to add
	 */
	public void addDirection(Direction direction) {
		this.mDirections.add(direction);
	}
	
	/**
	 * @return the mDirections
	 */
	public ArrayList<Direction> getmDirections() {
		return mDirections;
	}
	
	/**
	 * @param mDirections
	 *            the mDirections to set
	 */
	public void setmDirections(ArrayList<Direction> mDirections) {
		this.mDirections = mDirections;
	}
	
	/**
	 * @return a list of partition id that one can leave through the door
	 */
	public ArrayList<Integer> getD2PLeave(){
		ArrayList<Integer> result = new ArrayList<Integer>();
		
		for (int i = 0; i < this.mDirections.size(); i ++) {
			if (!result.contains(this.mDirections.get(i).getsID())) result.add(this.mDirections.get(i).getsID());
		}
		
		return result;
	}
	
	/**
	 * @return a list of partition id that one can enter through the door
	 */
	public ArrayList<Integer> getD2PEnter(){
		ArrayList<Integer> result = new ArrayList<Integer>();
		
		for (int i = 0; i < this.mDirections.size(); i ++) {
			if (!result.contains(this.mDirections.get(i).geteID())) result.add(this.mDirections.get(i).geteID());
		}
		
		return result;
	}
	
	public double nearestClosedTime(double t) {
		return this.cTime;
	}
	
	public double nearestOpenedTime(double t) {
		return this.oTime;
	}
	
	public boolean isInTimeInterval(double t) {
		if (this.oTime <= t && t < this.cTime) return true;
		else return false;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Comparable#compareTo(java.lang.Object)
	 */
	@Override
	public int compareTo(Object arg0) {
		// TODO Auto-generated method stub
		Door otherdoor = (Door) arg0;
		return this.mID > otherdoor.mID ? 1 : (this.mID == otherdoor.mID ? 0
				: -1);
	}
	
	/**
	 * toString
	 * 
	 * @return mID+x+y+mFloor+mPars
	 */
	public String toString() {
		String outputString = this.getmID() + "\t" + this.getX() + "\t"
				+ this.getY() + "\t" + this.getmFloor() + "\t";

		if (this.mPartitions != null) {
			Iterator<Integer> itr = this.mPartitions.iterator();

			while (itr.hasNext()) {
				outputString = outputString + itr.next() + "\t";
			}
		}

		return outputString;
	}
}
