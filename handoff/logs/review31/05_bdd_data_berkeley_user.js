/* global AWSCognito alertify agreeCount */
/* exported downloadFile downloadChallengeFile publish listPublicBenchmarks
* getSubmissionDetail countWeeklySubmission */

const weeklySubmissionLimit = 10;
let poolData = {
    UserPoolId: 'us-west-2_xVKg7XuzR',
    ClientId: '1kf3n8ivorpo46o9k39c63530b',
};
let userPool =
    new AWSCognito.CognitoIdentityServiceProvider.CognitoUserPool(poolData);

$('#confirm').on('change', function() {
    if ($('#password').val() != $('#confirm').val()) {
        $('#err-message').html('Two password should be the same');
    } else {
        $('#err-message').html('');
        let pw = $('#password').val();
        if (!(/[0-9]/.test(pw) && /[a-z]/.test(pw) &&
            /[A-Z]/.test(pw) && pw.length >= 8)) {
            $('#err-message').html(`Password must contains at least one digit
                <br>at least one lowercase character<br> at least one
                uppercase character <br>at least length of 8`);
        } else {
            $('#err-message').html('');
        }
    }
});

$('#register-button').click(register);
$('#login-button').click(login);
$('.signout-button').click(signout);
$('#agree-license-btn').click(agreeLicense);
$('#reset-button').click(resetPassword);
$('#changePassword-button').click(changePassword);
$('#download-section').click(downloadClick);
$('#sendCode-button').click(sendCode);
$('#organization-update').click(updateOrganization);
$('#upload-form').on('submit', uploadData);


/**
 * Register a user with AWS Congito
 */
function register() {
    let name = $('#name').val();
    let email = $('#email').val();
    let username = $('#email').val();
    let password = $('#password').val();
    let confirm = $('#confirm').val();
    let organization = $('#organization').val();

    if (name.trim() == '') {
        $('#err-message').html('Your name is required');
        // console.error('Your name is required');
        return;
    }

    // check email
    /* eslint-disable max-len */
    let re = /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    /* eslint-enable max-len */
    if (!re.test(email)) {
        $('#err-message').html('Please provide correct email address');
        // console.error('Please provide correct email address');
        return;
    }

    if (password != confirm) {
        $('#err-message').html('Two passwords do not match');
        // console.error('Two password not match');
        return;
    }

    let attributeList = [];

    let dataEmail = {
        Name: 'email',
        Value: email,
    };

    let dataName = {
        Name: 'name',
        Value: name,
    };
    let downloadCount = {
        Name: 'custom:download_count',
        Value: '0',
    };
    let dataOrganizaion = {
      Name: 'custom:organization',
      Value: organization,
    };

    let attributeEmail = new AWSCognito.
        CognitoIdentityServiceProvider.CognitoUserAttribute(dataEmail);
    let attributeName = new AWSCognito.
        CognitoIdentityServiceProvider.CognitoUserAttribute(dataName);
    let attributeDownloadCount = new AWSCognito.
        CognitoIdentityServiceProvider.CognitoUserAttribute(downloadCount);
    let attributeOrganizaion = new AWSCognito.
        CognitoIdentityServiceProvider.CognitoUserAttribute(dataOrganizaion);

    attributeList.push(attributeEmail);
    attributeList.push(attributeName);
    attributeList.push(attributeDownloadCount);
    attributeList.push(attributeOrganizaion);

    userPool.signUp(username, password, attributeList, null,
                            function(err) { // function can take in a parameter
        if (err) {
            $('#err-message').html(err.message);
            return;
        }
        // console.log("Success");
        // let cognitoUser = result.user;
        $('input').val('');
        $('#err-message').html('');
        $('#email-message').html('');
        alertify.alert('Please confirm registration in your email', function() {
            window.location.href='login.html';
        });
        // console.log('user name is ' + cognitoUser.getUsername());
    });
}

/**
 * login with AWS Cognito
 */
function login() {
    let username = $('#email').val();
    let password = $('#password').val();
    let authenticationData = {
        Username: username,
        Password: password,
    };
    let authenticationDetails = new AWSCognito.CognitoIdentityServiceProvider.
        AuthenticationDetails(authenticationData);
    let userData = {
        Username: username,
        Pool: userPool,
    };
    let cognitoUser = new AWSCognito.
        CognitoIdentityServiceProvider.CognitoUser(userData);
        cognitoUser.authenticateUser(authenticationDetails, {
        onSuccess: function() { // function can take in a parameter
            // console.log("sign in done");
            // console.log('access token + '
            // + result.getAccessToken().getJwtToken());
            /* Use the idToken for Logins Map when Federating User Pools with
            Cognito Identity or when passing through an Authorization Header
            to an API Gateway Authorizer
            */
            // console.log('idToken + ' + result.idToken.jwtToken);
            window.location.href='./portal.html';
        },

        onFailure: function(err) {
            // console.error(err.message);
            $('#err-message').html(err.message);
        },
    });
}

/**
 * Signout with AWS Cognito
 */
function signout() {
    let cognitoUser = userPool.getCurrentUser();

    if (cognitoUser != null) {
        // console.log("Sign out");
        cognitoUser.signOut();
        // cognitoUser.getSession(function(err, session) {
        //     if (err) {
        //         alert(err);
        //         return;
        //     }
        //     console.log('session validity: ' + session.isValid());
        // });
        window.location.href = './index.html';
    }
}

/**
 *
 * @return {boolean} check whether current user has login
 */
function checkLogin() {
    let cognitoUser = userPool.getCurrentUser();
    return cognitoUser != null;
}

/**
 *
 * @param {string} filename
 */
function downloadFile(filename) {
    if (checkLogin()) {
        // console.log("download");
        let data = {func: 'create_download_link', filename: filename};
        $.ajax({
            type: 'GET',
            url: 'https://2x5kv9t5uf.execute-api.' +
                    'us-west-2.amazonaws.com/production',
            data: data,
            success: function(result) {
                let link = document.createElement('a');
                link.download = 'test.txt';
                link.href = result;
                link.click();
            },
        });
    } else {
        window.location.href='./login.html';
    }
}

/**
 *
 * @param {string} event
 * @param {string} filename
 */
function downloadChallengeFile(event, filename) {
    if (checkLogin()) {
        let data = {func: 'create_download_challenge_link', filename: filename};
        $.ajax({
            type: 'GET',
            url: 'https://2x5kv9t5uf.execute-api.' +
                    'us-west-2.amazonaws.com/production',
            data: data,
            success: function(result) {
                let link = document.createElement('a');
                link.download = 'test.txt';
                link.href = result;
                link.dispatchEvent(new MouseEvent(`click`,
                    {bubbles: true, cancelable: true, view: window}));
                link.click();
            },
        });
    } else {
        window.location.href='./login.html';
    }
}

/**
 *
 * @param {event} e
 */
function uploadData(e) {
    e.preventDefault();
    // get the reference to the actual file in the input
    let formFile = $('#uploaded_data').get()[0].files[0];
    if (checkLogin()) {
        $('#submit-button').prop('disabled', true);
        let empty = false;
        $('#upload-form :input:not(:button)').each(function() {
            if ($(this).val() == '') {
                empty = true;
            }
        });
        if (empty) {
            $('#form-message').html('Please fill in the form!');
            $('#submit-button').prop('disabled', false);
            return;
        }
        let count = parseInt($('#weekly-submission-count').text());
        if (count <= 0) {
            $('#form-message').html('Exceeded weekly submission limit.');
            $('#submit-button').prop('disabled', false);
            return;
        }

        $('#form-message').html('');
        let filename = $('#uploaded_data').val().replace(/.*[/\\]/, '');
        let author = $('#name').html();
        let email = $('#email').html();
        let formData = {};
        let arr = $('#upload-form :input').serializeArray();
        for (let x of arr) {
            formData[x.name] = x.value;
        }
        formData['author'] = author;
        formData['filename'] = filename;
        formData['email'] = email;
        formData['func'] = 'create_upload_link';
        $.ajax({
            url: 'https://2x5kv9t5uf.execute-api.' +
                    'us-west-2.amazonaws.com/production',
            type: 'POST',
            data: JSON.stringify(formData),
            // Tell jQuery not to process data or worry about content-type
            // You *must* include these options!
            cache: false,
            dataType: 'json',
            contentType: 'application/json',
            processData: false,
            success: function(result) {
                // result is s3 signed url
                submitData(result, formFile);
            },
        });
    } else {
        window.location.href='./login.html';
    }
}

/**
 *
 * @param {string} uploadURL
 * @param {file} formFile
 * @param {object} formData
 */
function submitData(uploadURL, formFile) {
    // console.log(formFile);
    if (checkLogin()) {
        $.ajax({
            xhr: function() {
                let xhr = new window.XMLHttpRequest();

                xhr.upload.addEventListener('progress', function(e) {
                    if (e.lengthComputable) {
                        let percentComplete = e.loaded / e.total;
                        percentComplete = parseInt(percentComplete * 100);
                        $('#progress-bar')
                        .attr('aria-valuenow', percentComplete)
                        .css('width', percentComplete + '%')
                        .text(percentComplete + '%');
                    }
                }, false);
                return xhr;
            },
            type: 'PUT',
            url: uploadURL,
            data: formFile,
            // Tell jQuery not to process data or worry about content-type
            // You *must* include these options!
            // cache: false,
            contentType: 'binary/octet-stream',
            processData: false,
            success: function() { // function can take in a parameter
                // console.log(result);
                $('#submit-button').prop('disabled', false);
                $('#progress-bar').text('Uploaded Successfully!');
                $('form').trigger('reset');
                listMyFile(agreeCount);

                let count = parseInt($('#weekly-submission-count').text());
                $('#weekly-submission-count').html(count - 1);
            },
            error: function() {
                alert('Upload failed.');
            },
        });
    } else {
        window.location.href='./login.html';
    }
}

/**
 * function that handle download click on index.html
 */
function downloadClick() {
    if (checkLogin()) {
        window.location.href='./portal.html';
    } else {
        window.location.href='./login.html';
    }
}

/**
 * send verification code
 */
function sendCode() {
    let username = $('#email').val();
    let cognitoUser = new AWSCognito.CognitoIdentityServiceProvider.
        CognitoUser({
        Username: username,
        Pool: userPool,
    });
    cognitoUser.forgotPassword({
        onSuccess: function() { // function can take in a parameter
            // successfully initiated reset password request
            //   console.log('CodeDeliveryData from forgotPassword: ' + data);
              alert('Verification Code was sent to your email.');
        },
        onFailure: function(err) {
            alert(err);
        },
    });
}

/**
 * reset password with AWS Cognito
 */
function resetPassword() {
    let username = $('#email').val();
    let cognitoUser = new AWSCognito.CognitoIdentityServiceProvider.
        CognitoUser({
        Username: username,
        Pool: userPool,
    });
    let verificationCode = $('#code').val();
    let newPassword = $('#newPassword').val();
    cognitoUser.confirmPassword(verificationCode, newPassword, {
        onFailure(err) {
            // console.error(err.message);
             $('#err-message').html(err.message);
        },
        onSuccess() {
            window.location.href = './login.html';
        },
    });
}

/**
 * change password with AWS Cognito
 */
function changePassword() {
    let cognitoUser = userPool.getCurrentUser();
    let oldPassword = $('#oldPassword').val();
    let newPassword = $('#newPassword').val();
    if (cognitoUser != null) {
        cognitoUser.getSession(function(err) { // function can take in a param
            if (err) {
                alert(err);
                return;
            }
            // console.log('session validity: ' + session.isValid());

            cognitoUser.changePassword(oldPassword, newPassword,
                                    function(err) {
                if (err) {
                    $('#err-message').html(err.message);
                    return;
                }
                $('#err-message').html('');
                window.location.href='./portal.html';
            });
        });
    }
}

/**
 * update Organization in profile
 */
function updateOrganization() {
  let cognitoUser = userPool.getCurrentUser();
  if (cognitoUser == null) {
    window.location.href='./login.html';
  }
  let attributeList = [];
  let attribute = {
    Name: 'custom:organization',
    Value: $('#organization').val(),
  };
  attribute = new AWSCognito.CognitoIdentityServiceProvider.
      CognitoUserAttribute(attribute);
  attributeList.push(attribute);
  cognitoUser.getSession(function(err) { // function can take in a parameter
    if (err) {
      alert(err);
      return;
    }
    cognitoUser.updateAttributes(attributeList, function(err) {
      // function can take in a parameter
      if (err) {
        alert(err);
        return;
      }
    });
  });
}

/**
 * Agree license
 */
function agreeLicense() {
    let cognitoUser = userPool.getCurrentUser();
    if (cognitoUser == null) {
        window.location.href='./login.html';
    }
    if (!$('#license-checkbox').is(':checked')) {
        alert('Please agree to the license');
        return;
    }
    let attributeList = [];
    let attribute = {
        Name: 'custom:download_count',
        Value: agreeCount == 0 ? '1' : agreeCount,
    };
    attribute = new AWSCognito.CognitoIdentityServiceProvider.
        CognitoUserAttribute(attribute);
    attributeList.push(attribute);
    cognitoUser.getSession(function(err) { // function can take in a parameter
        if (err) {
            alert(err);
            return;
        }
        // console.log('session validity: ' + session.isValid());
        cognitoUser.updateAttributes(attributeList, function(err) {
            // function can take in a parameter
            if (err) {
                alert(err);
                return;
            }
            $('#license-checkbox').prop('disabled', true);
            $('#agree-license-btn').prop('disabled', true);
            $('.download-btn').prop('disabled', false);
            $('#submit-button').prop('disabled', false);
            $('#upload-form :input:not(:button)').each(function() {
                $(this).prop('disabled', false);
            });
            getDownloadInfo();
        });
    });
}

/**
 *  list public benchmarks
 */
function listPublicBenchmarks() {
    let attachData = {func: 'list_public_benchmarks'};
    $.ajax({
        type: 'GET',
        url: 'https://2x5kv9t5uf.execute-api.' +
                'us-west-2.amazonaws.com/production',
        data: attachData,
        success: function(result) {
            if (!(result.Items instanceof Array)) {
                return;
            }
            let data = [];
            for (let row of result.Items) {
                data.push([row.author, row.submission,
                                    row.uploadTime, row.grade]);
            }
          $('#myTable').dataTable({
            'data': data,
            'deferRender': true,
            'order': [[3, 'desc']],
          });
        },
    });
}

/**
 * list my file in s3
 * @param {int} downloadCount: 0 for download disabled, 1 for enabled
 */
function listMyFile(downloadCount) {
    if (checkLogin()) {
        // console.log("download");
        let author = $('#name').html();
        let attachData = {func: 'list_s3_files_by_user', author: author};
        $.ajax({
            type: 'GET',
            url: 'https://2x5kv9t5uf.execute-api.' +
            'us-west-2.amazonaws.com/production',
            data: attachData,
            success: function(result) {
                // console.log(result);
                $('#submission-table tbody').empty();
                let items = result.Items;
                items.sort(function(a, b) {
                    return new Date(b.uploadTime) - new Date(a.uploadTime);
                });
                for (let file of items) {
                    let tempName = file.file.split('/').pop();
                    // let filename = '"' + file.file + '"';

                    let grade = 0;
                    let details;
                    if (!file.grade) {
                        grade = 'processing';
                    } else if (file.breakdown == 'None') {
                        grade = file.grade;
                    } else {
                        grade = `<a data-toggle="tooltip"
                                    data-placement="right"
                               title="${file.breakdown}">${file.grade}</a>`;
                    }
                    if (file.grade && file.submissionId) {
                      let link = './submission_detail.html?' +
                          'submissionId=' + file.submissionId;
                      details = `<a type="button" style="margin:auto"
                                href=${link} class="btn btn-primary btn-xs">
                                Detail</button>`;
                    }
                    let submissionName = file.submission;
                    let timeStr = new Date(file.uploadTime).toLocaleString();

                    $('#submission-table tbody').append(`<tr>
                        <td>${submissionName}</td>
                        <td>${file.task}</td>
                        <td>${tempName}</td>
                        <td nowrap="nowrap">${timeStr}</td>
                        <td>${grade}</td>
                        <td vertical-align: "middle">${details}</td>
                    </tr>`);
                    $('#' + file.uploadTime).prop('disabled',
                        grade == 'processing' || file.breakdown == 'None');
                }
                $('.download-btn').prop('disabled', downloadCount == '0');
            },
        });
    } else {
        window.location.href='./login.html';
    }
}

/**
 * publish this submission to public benchmark
 * @param {string} filename
 */
function publish(filename) {
    if (checkLogin()) {
        let author = $('#name').html();
        let data = {func: 'publish', filename: filename, author: author};
        $.ajax({
            type: 'GET',
            url: 'https://2x5kv9t5uf.execute-api.' +
            'us-west-2.amazonaws.com/production',
            data: data,
            success: function() {
            },
        });
    } else {
        window.location.href='./login.html';
    }
}

/**
 * Invoke AWS Lambda function to get the download part if user agree license
 */
function getDownloadInfo() {
    if (checkLogin()) {
        let attachData = {func: 'get_download_info'};
        $.ajax({
            type: 'GET',
            url: 'https://2x5kv9t5uf.execute-api.' +
            'us-west-2.amazonaws.com/production',
            data: attachData,
            success: function(result) {
                $('#download-info').append(result);
                $('#paper-toggle').click(togglePaper);
                $('#challenge-toggle').click(toggleChallenges);
                $('#attention-toggle').click(toggleAttention);
            },
        });
    } else {
        window.location.href='./login.html';
    }
}

/**
 * paper toggle for plus minus button
 */
function togglePaper() {
    if ($('#cvpr-paper-link > .glyphicon-plus').length > 0) {
        $('#cvpr-paper-link').empty();
        $('#cvpr-paper-link').append('<span class=\'glyphicon ' +
                                        'glyphicon-minus\'></span>');
        $('#paper-table').show();
    } else {
        $('#cvpr-paper-link').empty();
        $('#cvpr-paper-link').append('<span class=\'glyphicon ' +
                                        'glyphicon-plus\'></span>');
        $('#paper-table').hide();
    }
}

/**
 * challenge toggle for plus minus button
 */
function toggleChallenges() {
    if ($('#cvpr-challenge-link > .glyphicon-plus').length > 0) {
        $('#cvpr-challenge-link').empty();
        $('#cvpr-challenge-link').append('<span class=\'glyphicon ' +
                                            'glyphicon-minus\'></span>');
        $('#challenge-table').show();
    } else {
        $('#cvpr-challenge-link').empty();
        $('#cvpr-challenge-link').append('<span class=\'glyphicon ' +
                                            'glyphicon-plus\'></span>');
        $('#challenge-table').hide();
    }
}

/**
 * Attention toggle for plus minus button
 */
function toggleAttention() {
    if ($('#attention-link > .glyphicon-plus').length > 0) {
        $('#attention-link').empty();
        $('#attention-link').append('<span class=\'glyphicon ' +
                                            'glyphicon-minus\'></span>');
        $('#attention-table').show();
    } else {
        $('#attention-link').empty();
        $('#attention-link').append('<span class=\'glyphicon ' +
                                            'glyphicon-plus\'></span>');
        $('#attention-table').hide();
    }
}

/**
 * challenge toggle for plus minus button
 * @param {string} username
 * @param {string} submissionNum
 */
function getSubmissionDetail(username, submissionNum) {
  if (checkLogin()) {
    let attachData = {func: 'list_s3_files_by_user', author: username};
    $.ajax({
      type: 'GET',
      url: 'https://2x5kv9t5uf.execute-api.' +
      'us-west-2.amazonaws.com/production',
      data: attachData,
      success: function(result) {
        // find entry with submissionNumber
        let items = result.Items;
        let submissionObj = items.find((o) => o.submissionId === submissionNum);
        if (!submissionObj) {
          alertify.alert('Sorry! You do not have this submission',
              function() {
            window.location.href='portal.html';
          });
        }
        let filepath = submissionObj['file'];
        let data = {
          func: 'get_submission_detail',
          author: username,
          filename: filepath,
        };
        $.ajax({
          type: 'GET',
          url: 'https://2x5kv9t5uf.execute-api.' +
          'us-west-2.amazonaws.com/production',
          data: data,
          success: function(result) {
            $('#submissionName').html(result.submission);
            $('#teamName').html(result.teamname);
            $('#task').html(result.task);
            $('#description').html(result.description);
            $('#score')
                .html(result.grade);
            $('#breakdown').html(result.breakdown);
          },
        });
      },
    });
  } else {
    window.location.href='./login.html';
  }
}


/**
 * Check number of submissions within a week in s3
 */
function countWeeklySubmission() {
    if (checkLogin()) {
        let author = $('#name').html();
        let attachData = {func: 'list_s3_files_by_user', author: author};
        $.ajax({
            type: 'GET',
            url: 'https://2x5kv9t5uf.execute-api.' +
                'us-west-2.amazonaws.com/production',
            data: attachData,
            success: function(result) {
                let counter = 0;
                let weekInMillisec = 604800000;
                let current = new Date();
                let items = result.Items;
                for (let file of items) {
                    let uploadTime = new Date(file.uploadTime);
                    let timeDiff = current.getTime() - uploadTime.getTime();
                    if (timeDiff <= weekInMillisec) {
                        counter += 1;
                    }
                }
                $('#weekly-submission-limit').html(weeklySubmissionLimit);
                $('#weekly-submission-count').html(
                                        weeklySubmissionLimit - counter);
            },
        });
    }
}
