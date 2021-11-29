package datagenerate;

import indoor_entitity.IndoorSpace;
import indoor_entitity.Partition;
import utilities.DataGenConstant;
import utilities.DoorType;
import utilities.RoomType;

import java.util.ArrayList;

public class ClassifyPartition {

	// constructor
	public ClassifyPartition() {
	}

	/**
	 * classify partitions according to topology properties
	 * 
	 * crucialpass: partition with 4 or more than 4 doors
	 */
	public void classifyPar() {
		int crucial = 0;
		int simple = 0;


		for (int i = 0; i < IndoorSpace.iPartitions.size(); i++) {
			Partition par = IndoorSpace.iPartitions.get(i);
			if (par.getmDoors().size() > 6) {
				par.settType(RoomType.CRUCIALPASS);
				IndoorSpace.iCrucialPartitions.add(par);
			}
			else {
				par.settType(RoomType.SIMPLEPASS);
			}

		}

//
	}
}
