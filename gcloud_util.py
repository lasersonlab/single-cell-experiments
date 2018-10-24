import os
import time

import googleapiclient.discovery


def list_instances(compute, project, zone):
    result = compute.instances().list(project=project, zone=zone).execute()
    return result["items"] if "items" in result else None


def create_instance(compute, project, zone, name):
    # Get the latest Debian Jessie image.
    image_response = (
        compute.images()
        .getFromFamily(project="debian-cloud", family="debian-9")
        .execute()
    )
    source_disk_image = image_response["selfLink"]

    # Configure the machine
    machine_type = "zones/%s/machineTypes/n1-standard-1" % zone
    # startup_script = open(
    #     os.path.join(
    #         os.path.dirname(__file__), 'startup-script.sh'), 'r').read()
    # image_url = "http://storage.googleapis.com/gce-demo-input/photo.jpg"
    # image_caption = "Ready for dessert?"

    config = {
        "name": name,
        "machineType": machine_type,
        # Specify the boot disk and the image to use as a source.
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "initializeParams": {
                    "sourceImage": source_disk_image,
                    "diskSizeGb": "40",
                },
            }
        ],
        # Specify a network interface with NAT to access the public
        # internet.
        "networkInterfaces": [
            {
                "network": "global/networks/default",
                "accessConfigs": [{"type": "ONE_TO_ONE_NAT", "name": "External NAT"}],
            }
        ],
        # Allow the instance to access cloud storage and logging.
        "serviceAccounts": [
            {
                "email": "default",
                "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
            }
        ],
        # Metadata is readable from the instance and allows you to
        # pass configuration from deployment scripts to instances.
        # 'metadata': {
        #     'items': [{
        #         # Startup script is automatically executed by the
        #         # instance upon startup.
        #         'key': 'startup-script',
        #         'value': startup_script
        #     }, {
        #         'key': 'url',
        #         'value': image_url
        #     }, {
        #         'key': 'text',
        #         'value': image_caption
        #     }, {
        #         'key': 'bucket',
        #         'value': bucket
        #     }]
        # }
    }

    return compute.instances().insert(project=project, zone=zone, body=config).execute()


def delete_instance(compute, project, zone, name):
    return (
        compute.instances().delete(project=project, zone=zone, instance=name).execute()
    )


def wait_for_operation(compute, project, zone, operation):
    print("Waiting for operation to finish...")
    while True:
        result = (
            compute.zoneOperations()
            .get(project=project, zone=zone, operation=operation)
            .execute()
        )

        if result["status"] == "DONE":
            print("done.")
            if "error" in result:
                raise Exception(result["error"])
            return result

        time.sleep(1)
